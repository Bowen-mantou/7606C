"""MLP noise predictor migrated from VariDP.

Ported from ``VariDP/dp/backbones.py`` (``MLPNoisePred``), the local baseline
of the VariDP state experiments. The donor received a precomputed time
embedding; here the sinusoidal embedding moved inside the backbone so every
``NoisePredictor`` gets the same raw timestep, and the flattened observation
features produced by the shared encoder replace the donor's private observation
MLP (this adapter never re-encodes RGB or proprioception).

Network shape (identical to the donor)::

    [flatten(noisy_actions) ‖ t_emb ‖ flatten(obs_features)]
        -> [hidden] * n_layers
        -> flatten(actions)

with Mish + LayerNorm between hidden layers and a linear output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PolicyConfig
from .base import NoisePredictor
from .timestep import SinusoidalPosEmb, expand_timesteps


class MLPBackbone(NoisePredictor):
    """Flatten ``(B, To, Dobs)`` into the MLP global conditioning vector.

    Structure comes from ``PolicyConfig`` (``mlp_*`` fields in
    ``baseline.toml``, donor defaults: 3 hidden layers of 256, time embedding
    128). The shared observation sequence is consumed unchanged.
    """

    def __init__(self, policy_cfg: PolicyConfig, *, obs_dim: int, action_dim: int):
        super().__init__()
        self.pred_horizon = policy_cfg.pred_horizon
        self.action_dim = action_dim
        self.time_embed = SinusoidalPosEmb(policy_cfg.mlp_time_embed_dim)
        flat_action_dim = policy_cfg.pred_horizon * action_dim
        flat_condition_dim = policy_cfg.obs_horizon * obs_dim
        dims = [flat_action_dim + policy_cfg.mlp_time_embed_dim + flat_condition_dim]
        dims += [policy_cfg.mlp_hidden_dim] * policy_cfg.mlp_layers
        dims += [flat_action_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers += [nn.Mish(), nn.LayerNorm(dims[index + 1])]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        if obs_features.ndim != 3:
            raise ValueError(
                f"expected observation features (B, To, Dobs), got {tuple(obs_features.shape)}"
            )
        batch = noisy_actions.shape[0]
        timesteps = expand_timesteps(timestep, batch, noisy_actions.device)
        condition = torch.cat(
            (
                noisy_actions.flatten(start_dim=1),
                self.time_embed(timesteps),
                obs_features.flatten(start_dim=1),
            ),
            dim=-1,
        )
        return self.net(condition).reshape(batch, self.pred_horizon, self.action_dim)
