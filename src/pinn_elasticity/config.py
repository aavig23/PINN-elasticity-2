"""Experiment configuration: YAML files -> validated, frozen dataclasses.

A config file may contain ``extends: <path>`` (relative to that file). The
extended file is loaded first and the extending file is deep-merged on top:
mappings merge key by key, everything else is replaced. After merging, every
field is required, unknown keys are rejected, and the material, geometry and
load values are locked to PINN_ELASTICITY_SPEC.md.
"""

import copy
import hashlib
import json
import math
import types
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Invalid or inconsistent configuration."""


# Spec §1.1 (L, H), §1.2 (E, nu, rho), §2.5 (P0, force per unit thickness).
# Experiments may not change these; a different value needs a spec revision.
SPEC_PHYSICS = {"L": 1.0, "H": 0.5, "E": 200.0e9, "nu": 0.30, "rho": 7800.0, "P0": 1000.0}

LOSS_GROUPS = ("pde", "free", "load", "clamp", "ic")


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    name: str
    parent: str | None
    stage: int
    seed: int


@dataclass(frozen=True)
class PhysicsConfig:
    L: float
    H: float
    E: float
    nu: float
    rho: float
    P0: float


@dataclass(frozen=True)
class TimeConfig:
    T_hat: float
    t_r_hat: float


@dataclass(frozen=True)
class ScalingConfig:
    mode: str


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: int
    width: int
    activation: str


@dataclass(frozen=True)
class ConstraintsConfig:
    mode: str


@dataclass(frozen=True)
class SamplingConfig:
    n_interior: int
    n_top: int
    n_bottom: int
    n_right: int
    n_left: int
    n_initial: int
    resample_every: int


@dataclass(frozen=True)
class LossConfig:
    weighting: str
    weights: dict[str, float]
    corner_exclusion_radius: float


@dataclass(frozen=True)
class OptimizerConfig:
    adam_lr: float
    adam_lr_final: float
    adam_iterations: int
    lbfgs_iterations: int
    lbfgs_history: int
    lbfgs_line_search: str


@dataclass(frozen=True)
class TrainingConfig:
    log_every: int
    lbfgs_log_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class PrecisionConfig:
    dtype: str


@dataclass(frozen=True)
class OutputConfig:
    nx: int
    nz: int
    n_snapshots: int
    snapshot_times: list[float] | None
    tip_history_points: int
    dpi: int


@dataclass(frozen=True)
class DiagnosticsConfig:
    edge_points: int
    edge_times: int
    precision_check_points: int


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    physics: PhysicsConfig
    time: TimeConfig
    scaling: ScalingConfig
    model: ModelConfig
    constraints: ConstraintsConfig
    sampling: SamplingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    precision: PrecisionConfig
    output: OutputConfig
    diagnostics: DiagnosticsConfig

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @property
    def config_hash(self) -> str:
        """SHA-256 of the fully resolved config (key order independent)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def snapshot_times(self) -> list[float]:
        """Stage 2 snapshot times t_hat. Default: n evenly spaced over (0, T_hat]."""
        if self.output.snapshot_times is not None:
            return list(self.output.snapshot_times)
        n, T = self.output.n_snapshots, self.time.T_hat
        return [T * k / n for k in range(1, n + 1)]


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load, merge (``extends`` chain, then ``overrides``), build and validate."""
    raw = _load_raw(Path(path).resolve(), ())
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = _build(Config, raw, "config")
    _validate(cfg)
    return cfg


def _load_raw(path: Path, chain: tuple[Path, ...]) -> dict:
    if path in chain:
        raise ConfigError(f"circular 'extends': {' -> '.join(str(p) for p in (*chain, path))}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _load_raw((path.parent / parent).resolve(), (*chain, path))
    return _deep_merge(base, data)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")
    return cls(**{n: _coerce(hints[n], data[n], f"{where}.{n}") for n in names})


def _coerce(tp, value, where: str):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(inner, value, where)
    if is_dataclass(tp):
        return _build(tp, value, where)
    if origin is dict:
        _, value_type = typing.get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return {str(k): _coerce(value_type, v, f"{where}.{k}") for k, v in value.items()}
    if origin is list:
        (item_type,) = typing.get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        return [_coerce(item_type, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint = ""
            if isinstance(value, str):
                hint = " (YAML reads exponents without a sign as text: write 2.0e+11, not 200.0e9)"
            raise ConfigError(f"{where}: expected a number, got {value!r}{hint}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported field type {tp}")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate(cfg: Config) -> None:
    for key, spec_value in SPEC_PHYSICS.items():
        value = getattr(cfg.physics, key)
        _check(
            math.isclose(value, spec_value, rel_tol=1e-12),
            f"physics.{key} = {value} differs from the spec value {spec_value}; "
            "material, geometry and P0 are locked to PINN_ELASTICITY_SPEC.md",
        )

    e = cfg.experiment
    _check(e.stage in (1, 2), "experiment.stage must be 1 (static) or 2 (dynamic)")
    _check(e.seed >= 0, "experiment.seed must be >= 0")

    _check(cfg.time.T_hat > 0, "time.T_hat must be > 0")
    _check(cfg.time.t_r_hat > 0, "time.t_r_hat must be > 0")
    _check(cfg.scaling.mode == "spec_default", "scaling.mode must be 'spec_default' (spec §3.2)")

    m = cfg.model
    _check(m.activation == "tanh", "model.activation must be 'tanh'")
    _check(m.hidden_layers >= 1 and m.width >= 1, "model.hidden_layers and model.width must be >= 1")

    mode = cfg.constraints.mode
    _check(mode in ("hard", "soft"), "constraints.mode must be 'hard' or 'soft'")

    s = cfg.sampling
    for name in ("n_interior", "n_top", "n_bottom", "n_right", "resample_every"):
        _check(getattr(s, name) > 0, f"sampling.{name} must be > 0")
    _check(s.n_left >= 0 and s.n_initial >= 0, "sampling.n_left and n_initial must be >= 0")
    if mode == "soft":
        _check(s.n_left > 0, "sampling.n_left must be > 0 when constraints.mode is 'soft'")
        if e.stage == 2:
            _check(s.n_initial > 0, "sampling.n_initial must be > 0 for a soft Stage 2 run")

    lo = cfg.loss
    _check(lo.weighting == "fixed", "loss.weighting must be 'fixed'")
    _check(set(lo.weights) == set(LOSS_GROUPS), f"loss.weights must have exactly the keys {list(LOSS_GROUPS)}")
    _check(all(w >= 0 for w in lo.weights.values()), "loss.weights must be >= 0")
    _check(lo.corner_exclusion_radius >= 0, "loss.corner_exclusion_radius must be >= 0")

    o = cfg.optimizer
    _check(o.adam_lr > 0 and o.adam_lr_final > 0, "optimizer learning rates must be > 0")
    _check(o.adam_iterations >= 0 and o.lbfgs_iterations >= 0, "optimizer iteration counts must be >= 0")
    _check(o.adam_iterations + o.lbfgs_iterations > 0, "at least one optimizer iteration is required")
    _check(o.lbfgs_history > 0, "optimizer.lbfgs_history must be > 0")
    _check(o.lbfgs_line_search == "strong_wolfe", "optimizer.lbfgs_line_search must be 'strong_wolfe'")

    t = cfg.training
    for name in ("log_every", "lbfgs_log_every", "checkpoint_every"):
        _check(getattr(t, name) > 0, f"training.{name} must be > 0")

    _check(cfg.precision.dtype in ("float32", "float64"), "precision.dtype must be 'float32' or 'float64'")

    out = cfg.output
    _check(out.nx >= 2 and out.nz >= 2, "output.nx and output.nz must be >= 2")
    _check(out.n_snapshots >= 1, "output.n_snapshots must be >= 1")
    if out.snapshot_times is not None:
        _check(
            len(out.snapshot_times) >= 1 and all(0 <= t <= cfg.time.T_hat for t in out.snapshot_times),
            "output.snapshot_times must lie in [0, T_hat]",
        )
    _check(out.tip_history_points >= 2, "output.tip_history_points must be >= 2")
    _check(out.dpi > 0, "output.dpi must be > 0")

    d = cfg.diagnostics
    _check(d.edge_points >= 3 and d.edge_times >= 2, "diagnostics.edge_points >= 3 and edge_times >= 2")
    _check(d.precision_check_points >= 1, "diagnostics.precision_check_points must be >= 1")
