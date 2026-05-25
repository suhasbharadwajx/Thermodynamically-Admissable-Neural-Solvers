import torch
import torch.nn as nn
import numpy as np

class PiezoelectricPINN(nn.Module):
    """
    Feedforward Network with Absolute Hard Constraints.
    Guarantees Dirichlet BCs and 2nd-Order Time ICs (Position & Velocity) analytically.
    """
    def __init__(self, in_dim=2, out_dim=2, hidden=180, layers=8):
        super().__init__()
        modules = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)
        
        # Xavier Uniform Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, amplitude_ratio):
        x_c, t_c = x[:, 0:1], x[:, 1:2]
        raw = self.net(x)
        u_raw, phi_raw = raw[:, 0:1], raw[:, 1:2]

        # Exact Distance Functions
        bc_basis = x_c * (1.0 - x_c)  
        t2 = t_c ** 2                 
        sx = torch.sin(np.pi * x_c)   

        # Topologically Constrained Ansatz
        u = (bc_basis * t2 * u_raw) + sx
        phi = (bc_basis * t2 * phi_raw) + (amplitude_ratio * sx)

        return torch.cat([u, phi], dim=1)
