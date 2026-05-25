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

# Evaluation
x_test = np.linspace(0, 1, 200)
t_test = np.linspace(0, 1, 200)
Xg, Tg = np.meshgrid(x_test, t_test)
XT = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
XT_t = torch.tensor(XT, dtype=torch.float32, device=device)

with torch.no_grad():
    pred = model(XT_t, amplitude_ratio).detach().cpu().numpy()
    exact = exact_sol(XT)

norm = lambda v: np.linalg.norm(v)
for idx, name in enumerate(['u', 'phi']):
    err = norm(pred[:, idx] - exact[:, idx]) / (norm(exact[:, idx]) + 1e-14)
    print(f"L2 Error {name}: {err:.5e}")

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history)
plt.savefig('loss_trajectory.png')
