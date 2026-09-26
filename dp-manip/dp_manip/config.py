"""Layered, typed experiment configuration for training and evaluation."""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar


@dataclass
class TaskConfig:
    name: str
    env_id: str
    control_mode: str
    max_episode_steps: int
    sim_backend: str
    shader_pack: str


@dataclass
class DataConfig:
    root: str
    train_path: str
    val_path: str
    num_demos: int
    val_num_demos: int


@dataclass
class VisionConfig:
    feature_dim: int
    random_shift: int
    share_camera_encoder: bool


@dataclass
class PolicyConfig:
    obs_horizon: int
    act_horizon: int
    pred_horizon: int
    diffusion_step_embed_dim: int
    unet_dims: list[int]
    kernel_size: int
    n_groups: int


@dataclass
class TrainConfig:
    seed: int
    total_iters: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    log_freq: int
    resume_freq: int
    validation_steps: list[int]
    checkpoint_steps: list[int]
    amp: bool


@dataclass
class EmaConfig:
    decay: float


@dataclass
class DiffusionConfig:
    num_diffusion_iters: int
    num_inference_iters: int


@dataclass
class EvalConfig:
    val_seed_start: int
    val_episodes: int
    test_seed_start: int
    test_episodes: int
    num_envs: int
    inference_seed: int


@dataclass
class Config:
    task: TaskConfig
    data: DataConfig
    vision: VisionConfig
    policy: PolicyConfig
    train: TrainConfig
    ema: EmaConfig
    diffusion: DiffusionConfig
    eval: EvalConfig

    def validate(self) -> None:
        if self.task.sim_backend != "physx_cpu":
            raise ValueError("evaluation must use physx_cpu to match the generated demonstrations")
        if self.data.num_demos not in {25, 50, 100, 200, 400}:
            raise ValueError("data.num_demos must be one of 25, 50, 100, 200, 400")
        if self.data.val_num_demos < 1:
            raise ValueError("data.val_num_demos must be positive")
        policy = self.policy
        if min(policy.obs_horizon, policy.act_horizon, policy.pred_horizon) < 1:
            raise ValueError("all horizons must be positive")
        if policy.obs_horizon + policy.act_horizon - 1 > policy.pred_horizon:
            raise ValueError("need obs_horizon + act_horizon - 1 <= pred_horizon")
        diffusion = self.diffusion
        if diffusion.num_diffusion_iters < 1 or diffusion.num_inference_iters < 1:
            raise ValueError("diffusion iteration counts must be positive")
        if diffusion.num_inference_iters > diffusion.num_diffusion_iters:
            raise ValueError("inference diffusion iterations cannot exceed training iterations")
        if self.vision.feature_dim < 1 or self.vision.random_shift < 0:
            raise ValueError("invalid vision encoder settings")
        train = self.train
        if min(train.total_iters, train.batch_size, train.log_freq, train.resume_freq) < 1:
            raise ValueError("training counts must be positive")
        for step in [*train.validation_steps, *train.checkpoint_steps]:
            if step < 1:
                raise ValueError(f"intermediate step {step} must be positive")
        if not 0.0 <= self.ema.decay < 1.0:
            raise ValueError("ema.decay must be in [0, 1)")
        evaluation = self.eval
        if evaluation.val_episodes < 1 or evaluation.test_episodes < 1 or evaluation.num_envs < 1:
            raise ValueError("evaluation counts must be positive")
        if set(self.val_seeds()) & set(self.test_seeds()):
            raise ValueError("validation and test rollout seed ranges overlap")

    def val_seeds(self) -> list[int]:
        return list(range(self.eval.val_seed_start, self.eval.val_seed_start + self.eval.val_episodes))

    def test_seeds(self) -> list[int]:
        return list(range(self.eval.test_seed_start, self.eval.test_seed_start + self.eval.test_episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_T = TypeVar("_T")
_SECTIONS = {
    "task": TaskConfig,
    "data": DataConfig,
    "vision": VisionConfig,
    "policy": PolicyConfig,
    "train": TrainConfig,
    "ema": EmaConfig,
    "diffusion": DiffusionConfig,
    "eval": EvalConfig,
}


def _section(cls: type[_T], raw: dict[str, Any], name: str) -> _T:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"config section {name!r} must be a table")
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"invalid or incomplete [{name}] section: {error}") from error


def _adapt_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map version-1 checkpoint config fields at the checkpoint boundary."""
    raw = copy.deepcopy(raw)
    policy = raw.get("policy")
    if isinstance(policy, dict):
        diffusion = raw.setdefault("diffusion", {})
        for name in ("num_diffusion_iters", "num_inference_iters"):
            if name in policy:
                value = policy.pop(name)
                if name in diffusion and diffusion[name] != value:
                    raise ValueError(f"conflicting legacy policy.{name} and diffusion.{name}")
                diffusion[name] = value
    train = raw.get("train")
    if isinstance(train, dict) and "ema_decay" in train:
        ema = raw.setdefault("ema", {})
        value = train.pop("ema_decay")
        if "decay" in ema and ema["decay"] != value:
            raise ValueError("conflicting legacy train.ema_decay and ema.decay")
        ema["decay"] = value
    return raw


def from_dict(raw: dict[str, Any]) -> Config:
    """Build a resolved config, adapting the version-1 checkpoint layout."""
    raw = _adapt_legacy_config(raw)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(**{name: _section(cls, raw, name) for name, cls in _SECTIONS.items()})
    cfg.validate()
    return cfg


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_legacy_task(path: Path, raw: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if set(raw) != {"legacy"}:
        return path, raw
    legacy = raw["legacy"]
    if not isinstance(legacy, dict) or set(legacy) != {"task_config"}:
        raise ValueError("legacy config must contain only legacy.task_config")
    task_path = (path.parent / legacy["task_config"]).resolve()
    return task_path, _read_toml(task_path)


def _default_baseline(task_path: Path) -> Path:
    candidates = (task_path.parent / "baseline.toml", task_path.parent.parent / "baseline.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find baseline.toml for {task_path}")


def load(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    baseline: str | Path | None = None,
) -> Config:
    """Resolve ``baseline + task + runtime overrides`` into one config."""
    task_path = Path(path).resolve()
    task_path, task_raw = _resolve_legacy_task(task_path, _read_toml(task_path))
    baseline_path = Path(baseline).resolve() if baseline is not None else _default_baseline(task_path)
    raw = _deep_merge(_read_toml(baseline_path), task_raw)
    for item in overrides:
        key, separator, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not separator or not dot or section not in _SECTIONS:
            raise ValueError(f"override must look like section.key=value: {item!r}")
        try:
            parsed = tomllib.loads(f"value = {value}")["value"]
        except tomllib.TOMLDecodeError:
            parsed = value
        raw.setdefault(section, {})[name] = parsed
    return from_dict(raw)
