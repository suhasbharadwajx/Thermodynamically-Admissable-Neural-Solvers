import torch
import numpy as np

# Physical Constants (Normalized)
rho = 1.0
c_E = 1.0
e_33 = 0.8
eps_S = 0.6
omega_exact = np.pi * np.sqrt((c_E + (e_33**2 / eps_S)) / rho)
amplitude_ratio = e_33 / eps_S

def exact_sol(x):
    X, t = x[:, 0:1], x[:, 1:2]
    u = np.sin(np.pi * X) * np.cos(omega_exact * t)
    phi = amplitude_ratio * np.sin(np.pi * X) * np.cos(omega_exact * t)
    return np.hstack([u, phi])

def piezo_residual(model, x):
    x_g = x.clone().detach().requires_grad_(True)
    out = model(x_g, amplitude_ratio)
    u, phi = out[:, 0:1], out[:, 1:2]

    u_x = torch.autograd.grad(u.sum(), x_g, create_graph=True)[0][:, 0:1]
    u_t = torch.autograd.grad(u.sum(), x_g, create_graph=True)[0][:, 1:2]
    phi_x = torch.autograd.grad(phi.sum(), x_g, create_graph=True)[0][:, 0:1]

    u_xx = torch.autograd.grad(u_x.sum(), x_g, create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t.sum(), x_g, create_graph=True)[0][:, 1:2]
    phi_xx = torch.autograd.grad(phi_x.sum(), x_g, create_graph=True)[0][:, 0:1]

    sigma_x = c_E * u_xx + e_33 * phi_xx
    D_x = e_33 * u_xx - eps_S * phi_xx

    r_mech = rho * u_tt - sigma_x
    r_elec = D_x 
    return r_mech, r_elec
