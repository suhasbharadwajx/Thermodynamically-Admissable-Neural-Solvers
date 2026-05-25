import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

from model import PiezoelectricPINN
from physics import piezo_residual, exact_sol, amplitude_ratio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Parameters
Nd = 20000
model = PiezoelectricPINN().to(device)
Xd_static = torch.rand(Nd, 2).to(device)
loss_history = []

def loss_fn(Xd):
    r_mech, r_elec = piezo_residual(model, Xd)
    return (r_mech**2).mean() + (r_elec**2).mean()

# Stage 1: Adam
opt_adam = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = CosineAnnealingLR(opt_adam, T_max=18000, eta_min=1e-5)

for ep in range(18000):
    opt_adam.zero_grad()
    loss = loss_fn(Xd_static)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt_adam.step()
    scheduler.step()
    loss_history.append(loss.item())

# Stage 2: AdamW
opt_adamw = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
for ep in range(12000):
    opt_adamw.zero_grad()
    loss = loss_fn(Xd_static)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt_adamw.step()
    loss_history.append(loss.item())

# Stage 3: L-BFGS
opt_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=600, 
    tolerance_grad=1e-10, tolerance_change=1e-10,
    history_size=80, line_search_fn='strong_wolfe'
)

def closure():
    opt_lbfgs.zero_grad()
    loss = loss_fn(Xd_static)
    loss.backward()
    loss_history.append(loss.item())
    return loss

opt_lbfgs.step(closure)

x_eval = np.linspace(0, 1, 200)
times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
labels = [r'$u(x,t)$', r'$\phi(x,t)$']
syms = ['u', r'\phi']

for i, (field_idx, title) in enumerate(zip([0, 1], ['Displacement', 'Electric Potential'])):
    for t in times:
        X_slice = np.column_stack([x_eval, np.full_like(x_eval, t)])
        X_tensor = torch.tensor(X_slice, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred = model(X_tensor, amplitude_ratio).detach().cpu().numpy()[:, field_idx]
            exact = exact_sol(X_slice)[:, field_idx]
            
        axs[0, i].plot(x_eval, pred, label=f't={t}')
        if t == 0.0:
            axs[0, i].plot(x_eval, exact, 'k--', alpha=0.5, label='Exact')
        
        axs[1, i].semilogy(x_eval, np.abs(pred - exact) + 1e-16, label=f't={t}')

    axs[0, i].set_title(f"{title} ({syms[i]}(x,t))")
    axs[0, i].set_xlabel("Position (x)")
    axs[1, i].set_title(f"Absolute Error |{syms[i]}_{{pred}} - {syms[i]}_{{exact}}|")
    axs[1, i].set_xlabel("Position (x)")

for ax in axs.flat:
    ax.grid(True, alpha=0.3)
axs[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure_2.png')
plt.show()

# Loss Trajectory Plot
plt.figure(figsize=(10, 5), dpi=300)
plt.semilogy(loss_history)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Optimization Dynamics')
plt.grid(True, alpha=0.3)
plt.savefig('loss_trajectory.png')
plt.show()
