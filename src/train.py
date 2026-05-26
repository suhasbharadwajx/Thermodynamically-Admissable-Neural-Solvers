import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

from model import PiezoelectricPINN
from physics import piezo_residual, exact_sol, amplitude_ratio, wave_speed, omega_exact

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

print(f"Rigorous Physics Validation:")
print(f"Stiffened Acoustic Wave Speed: {wave_speed:.4f}")
print(f"True Eigenfrequency (omega): {omega_exact:.4f} rad/s")
print(f"Potential Amplitude Ratio (e/eps_S): {amplitude_ratio:.4f}\n")

model = PiezoelectricPINN().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

Nd = 20000
Xd_cpu = torch.rand(Nd, 2)
batch_pde = 3000
loss_history = []

def loss_fn_pde(Xd_batch):
    r_mech, r_elec = piezo_residual(model, Xd_batch)
    return (r_mech**2).mean() + (r_elec**2).mean()


#Stage 1: Adam
print("STAGE 1: Adam with Cosine Annealing (18,000 epochs)")
opt_adam = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(opt_adam, T_max=18000, eta_min=1e-5)

for ep in range(18000):
    idx = torch.randperm(Nd)[:batch_pde]
    Xd = Xd_cpu[idx].to(device)
    
    opt_adam.zero_grad()
    loss = loss_fn_pde(Xd)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt_adam.step()
    scheduler.step()
    loss_history.append(loss.item())

    if (ep + 1) % 2000 == 0:
        print(f"Epoch {ep+1:5d}: Loss PDE = {loss.item():.4e} | LR = {scheduler.get_last_lr()[0]:.2e}")

torch.cuda.empty_cache()
gc.collect()

#Stage 2: AdamW
print("\nSTAGE 2: AdamW Fine-Tuning (12,000 epochs)")
opt_adamw = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for ep in range(12000):
    idx = torch.randperm(Nd)[:batch_pde]
    Xd = Xd_cpu[idx].to(device)
    
    opt_adamw.zero_grad()
    loss = loss_fn_pde(Xd)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt_adamw.step()
    loss_history.append(loss.item())

    if (ep + 1) % 2000 == 0:
        print(f"Epoch {ep+1:5d}: Loss PDE = {loss.item():.4e}")

torch.cuda.empty_cache()
gc.collect()


#Stage 3: L-BFGS
print("\nSTAGE 3: L-BFGS (Deterministic Data)")
Xd_lbfgs = Xd_cpu.to(device) 

opt_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, max_iter=600, tolerance_grad=1e-10, tolerance_change=1e-10,
    history_size=80, line_search_fn='strong_wolfe'
)

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    opt_lbfgs.zero_grad()
    loss = loss_fn_pde(Xd_lbfgs)
    loss.backward()
    lbfgs_iter += 1
    loss_history.append(loss.item())
    
    if lbfgs_iter % 50 == 0:
        print(f"L-BFGS iter {lbfgs_iter:3d}: Loss = {loss.item():.6e}")
    return loss

opt_lbfgs.step(closure)

#Final Evaluation & Visualization
print("\nFINAL EVALUATION")
x_test = np.linspace(0, 1, 450)
t_test = np.linspace(0, 1, 450)
Xg, Tg = np.meshgrid(x_test, t_test)
XT = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
XT_t = torch.tensor(XT, dtype=torch.float32, device=device)

with torch.no_grad():
    pred = model(XT_t, amplitude_ratio).detach().cpu().numpy()
    exact = exact_sol(XT)

norm = lambda v: np.linalg.norm(v)
for idx, name in enumerate(['u (displacement)', 'φ (electric potential)']):
    err = norm(pred[:, idx] - exact[:, idx]) / (norm(exact[:, idx]) + 1e-14)
    print(f"Global L2 Error - {name:25s}: {err:.5e}")

plt.rcParams.update({'font.size': 12})
times_plot = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
fig1, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=300)

for j, (field, sym) in enumerate(zip(['Displacement', 'Electric Potential'], ['u', r'\phi'])):
    ax1, ax2 = axs[0, j], axs[1, j]
    for tv in times_plot:
        Xline = np.column_stack([x_test, np.full_like(x_test, tv)])
        Xt_line = torch.tensor(Xline, dtype=torch.float32, device=device)
        pred_line = model(Xt_line, amplitude_ratio).detach().cpu().numpy()[:, j]
        ex_line = exact_sol(Xline)[:, j]
        
        ax1.plot(x_test, pred_line, label=f"t={tv}")
        ax1.plot(x_test, ex_line, 'k--', alpha=0.5, label='Exact' if tv == 0.0 else "")
        ax2.semilogy(x_test, np.abs(pred_line - ex_line) + 1e-16, label=f"t={tv}")

    ax1.set_title(f"{field} (${sym}(x,t)$)", fontsize=14, fontweight='bold')
    ax2.set_title(f"Absolute Error |${sym}_{{pred}} - {sym}_{{exact}}|$", fontsize=14)
    for ax in [ax1, ax2]:
        ax.set_xlabel("Position $x$")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_grid.png', format='png', bbox_inches='tight')

plt.rcParams.update({'font.size': 14})
fig2, ax = plt.subplots(figsize=(10, 6), dpi=300)
iterations = np.arange(len(loss_history))
ax.semilogy(iterations, loss_history, color='#1f77b4', linewidth=1.5, label=r'PDE Residual Loss $\mathcal{L}(\theta)$')

stage1_end = 18000
stage2_end = 18000 + 12000
ax.axvline(x=stage1_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=stage2_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

y_text = max(loss_history) * 0.5  
ax.text(stage1_end / 2, y_text, 'Stage 1\n(Adam)', horizontalalignment='center', fontweight='bold')
ax.text(stage1_end + (12000 / 2), y_text, 'Stage 2\n(AdamW)', horizontalalignment='center', fontweight='bold')
ax.text(stage2_end + (len(loss_history) - stage2_end) / 2, y_text, 'Stage 3\n(L-BFGS)', horizontalalignment='center', fontweight='bold')

ax.set_xlabel('Training Epochs / L-BFGS Evaluations', fontweight='bold')
ax.set_ylabel(r'Loss Magnitude (Log Scale)', fontweight='bold')
ax.set_title('Optimization Dynamics of the Constrained PINN', fontweight='bold', pad=15)
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend(loc='lower left', framealpha=1.0)

plt.tight_layout()
plt.savefig('loss_trajectory.png', format='png', bbox_inches='tight')
