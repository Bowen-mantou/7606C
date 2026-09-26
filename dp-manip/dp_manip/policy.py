"""RGB-conditioned Diffusion Policy built on the shared observation encoder."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .config import DiffusionConfig, PolicyConfig, VisionConfig
from .data import NormalizationStats
from .observation_encoder import ObservationEncoder


class DiffusionPolicy(nn.Module):
    """Encode RGB + proprioception and denoise a normalized action sequence."""

    def __init__(
        self,
        policy_cfg: PolicyConfig,
        vision_cfg: VisionConfig,
        diffusion_cfg: DiffusionConfig,
        *,
        image_shape: tuple[int, int, int],
        proprio_dim: int,
        action_dim: int,
        stats: NormalizationStats,
    ):
        super().__init__()
        self.obs_horizon = policy_cfg.obs_horizon
        self.act_horizon = policy_cfg.act_horizon
        self.pred_horizon = policy_cfg.pred_horizon
        self.action_dim = action_dim
        self.num_inference_iters = diffusion_cfg.num_inference_iters
        self.observation_encoder = ObservationEncoder(
            vision_cfg,
            obs_horizon=policy_cfg.obs_horizon,
            image_shape=image_shape,
            proprio_dim=proprio_dim,
            stats=stats,
        )
        # The conditional UNet consumes one flat FiLM vector per sample. That
        # flattening belongs to the backbone adapter, not to the observation
        # encoder, which always returns ``(B, To, Dobs)``.
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=policy_cfg.obs_horizon * self.observation_encoder.output_dim,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=policy_cfg.unet_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_cfg.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.register_buffer("action_low", torch.as_tensor(stats.action_low))
        self.register_buffer("action_high", torch.as_tensor(stats.action_high))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low

    def observation_features(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return shared observation features shaped ``(B, To, Dobs)``."""
        return self.observation_encoder(rgb, proprio)

    def flatten_observation(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Flatten ``(B, To, Dobs)`` features into UNet conditioning ``(B, To*Dobs)``."""
        return self.observation_encoder(rgb, proprio).flatten(start_dim=1)

    def compute_loss(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.flatten_observation(rgb, proprio)
        actions = self.normalize_action(actions.to(dtype=torch.float32))
        noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            generator=generator,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.noise_pred_net(noisy_actions, timesteps, global_cond=condition)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def get_action(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``(B, act_horizon, action_dim)`` in the environment's units."""
        condition = self.flatten_observation(rgb, proprio)
        sample = torch.randn(
            (rgb.shape[0], self.pred_horizon, self.action_dim),
            device=rgb.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_iters, device=rgb.device)
        # Unlike add_noise(), DDPMScheduler.step() does not move these tensors
        # to the sample device. A freshly loaded evaluation-only policy has not
        # called add_noise(), so move them explicitly before CUDA sampling.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(rgb.device)
        self.noise_scheduler.one = self.noise_scheduler.one.to(rgb.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.noise_pred_net(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(
                prediction, timestep, sample, generator=generator
            ).prev_sample
        start = self.obs_horizon - 1
        return self.unnormalize_action(sample[:, start : start + self.act_horizon])


def num_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def adapt_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Map pre-ObservationEncoder keys onto the current module layout.

    Inference checkpoints, resume checkpoints, and EMA shadows written before
    the encoder was extracted keep the camera weights under ``image_encoders.*``
    and the proprio buffers at the top level; version-1 checkpoints used the
    legacy ``state_*`` vocabulary.
    """
    adapted = state_dict.copy()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        adapted._metadata = metadata
    for legacy, canonical in (
        ("state_mean", "observation_encoder.proprio_mean"),
        ("state_std", "observation_encoder.proprio_std"),
        ("proprio_mean", "observation_encoder.proprio_mean"),
        ("proprio_std", "observation_encoder.proprio_std"),
    ):
        if legacy in adapted:
            if canonical in adapted:
                raise ValueError(f"checkpoint contains both {legacy!r} and {canonical!r}")
            adapted[canonical] = adapted.pop(legacy)
    for key in [key for key in adapted if key.startswith("image_encoders.")]:
        canonical = "observation_encoder." + key
        if canonical in adapted:
            raise ValueError(f"checkpoint contains both {key!r} and {canonical!r}")
        adapted[canonical] = adapted.pop(key)
    return adapted


def load_policy_state_dict(policy: DiffusionPolicy, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True):
    """Load a policy, adapting pre-ObservationEncoder checkpoint key names."""
    return policy.load_state_dict(adapt_legacy_state_dict(state_dict), strict=strict)
