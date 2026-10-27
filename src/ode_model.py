"""
Latent Neural ODE for malware concept-drift detection.

Architecture
------------
encoder(x)        -> z0      (initial latent state at t=0)
odeint(z0, t)     -> z(t)    (continuous-time evolution via adjoint method)
decoder(z(t))     -> p(malware)

The adjoint sensitivity method (torchdiffeq.odeint_adjoint) backpropagates
through the ODE solver in constant memory w.r.t. the number of function
evaluations, making training feasible on a single consumer GPU.

Solver choice: dopri5 (adaptive, default) or rk4 (fixed-step) for ablation.

Loss: Focal Loss (targets class-imbalance robustness on malware subsets).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint


class ODEFunc(nn.Module):
    """Neural network defining dz/dt = f(z, t). Time-invariant here."""

    def __init__(self, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        # small bias init so early dynamics are near-identity (stable start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, z):
        return self.net(z)


class LatentNeuralODE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden=64,
                 solver="dopri5", use_ode=True, atol=1e-3, rtol=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.solver = solver
        self.use_ode = use_ode
        self.atol = atol
        self.rtol = rtol

        # Decoder input = latent state (+ observation time when ODE is active)
        decoder_in = latent_dim + (1 if use_ode else 0)

        # Encoder: raw PE features -> initial latent state z(t0)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

        # Continuous-time dynamics (only used when use_ode=True)
        self.ode_func = ODEFunc(latent_dim, hidden=hidden)

        # Decoder: latent state -> malware logit
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, t):
        """
        x : (batch, input_dim)   features
        t : (batch,) or scalar   normalized time in [0, 1] when x was observed
        Returns logits (batch,), z0 (batch, latent), z_t (batch, latent)
        """
        z0 = self.encoder(x)

        if self.use_ode:
            # integrate each sample from t=0 to its observation time t_i
            t = t.to(x.device).float()
            # odeint expects a 1-D time tensor shared across the batch; to handle
            # per-sample times we integrate to the max time in the batch and then
            # re-evaluate is unnecessary -- instead integrate to each t_i via a
            # common grid [0, t_max] and take the state at t_max, then correct by
            # integrating z0 -> t_i individually is too costly. Simpler robust
            # approach: integrate to a fixed horizon T=1 for every sample, then
            # the decoder sees z(T). To retain time-awareness we append t to z.
            T = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
            z_traj = odeint_adjoint(self.ode_func, z0, T,
                                    method=self.solver,
                                    atol=self.atol, rtol=self.rtol)  # (2, B, L)
            z_t = z_traj[-1]  # state at horizon T=1
            # inject observation time so the decoder can condition on "when"
            z_in = torch.cat([z_t, t.unsqueeze(-1)], dim=-1)
        else:
            # Decoupled ablation: no ODE, classify from z0 directly
            z_t = z0
            z_in = z0

        logits = self.decoder(z_in).squeeze(-1)
        return logits, z0, z_t

    def latent_states(self, x, t):
        """Return z0 and evolved z_t for UMAP visualisation."""
        with torch.no_grad():
            logits, z0, z_t = self.forward(x, t)
        return z0, z_t


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # p_t
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
