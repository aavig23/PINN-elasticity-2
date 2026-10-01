"""Run folders: runs/EXPnnn/<UTC-timestamp>_stage<n>_seed<k>/ (spec §7.3-7.4)."""

from datetime import datetime, timezone
from pathlib import Path

FOLDER_LIMIT_MB = 20.0  # warn above this (spec §7.4)
CHECKPOINT_LIMIT_MB = 50.0  # never commit checkpoints above this (spec §7.4)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_run_id(stage: int, seed: int, now: datetime | None = None) -> str:
    now = now or utc_now()
    return f"{now:%Y%m%dT%H%M%SZ}_stage{stage}_seed{seed}"


def create_run_dir(root: Path, experiment_id: str, run_id: str) -> Path:
    """Create root/experiment_id/run_id; never reuse an existing folder (adds _2, _3, ...)."""
    base = Path(root) / experiment_id
    path, n = base / run_id, 1
    while path.exists():
        n += 1
        path = base / f"{run_id}_{n}"
    path.mkdir(parents=True)
    return path


def _mb(n_bytes: int) -> float:
    return n_bytes / 1e6


def size_report(run_dir: Path) -> dict:
    """Folder size, oversized checkpoints and warnings for a finished run."""
    run_dir = Path(run_dir)
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    oversized = sorted(
        p.relative_to(run_dir).as_posix()
        for p in files
        if p.suffix == ".pt" and _mb(p.stat().st_size) > CHECKPOINT_LIMIT_MB
    )
    warnings = []
    if _mb(total) > FOLDER_LIMIT_MB:
        warnings.append(f"run folder is {_mb(total):.1f} MB (target < {FOLDER_LIMIT_MB:.0f} MB)")
    for name in oversized:
        warnings.append(f"{name} is over {CHECKPOINT_LIMIT_MB:.0f} MB and must not be committed")
    return {"size_mb": round(_mb(total), 3), "oversized_checkpoints": oversized, "warnings": warnings}
