"""Start and resume experiment runs (plan §17, §19-20).

A full run checks the Git state, creates runs/EXPnnn/<run_id>/, writes
config.resolved.yaml and metadata.json, trains (the trainer writes
losses.csv, summary.json and checkpoints/), then records the outcome and
folder size. Smoke runs skip the Git checks and go to outputs/smoke/.
"""

from dataclasses import dataclass
from pathlib import Path

from ..config import load_config
from ..physics.scaling import Scales
from ..training.trainer import Trainer
from .git_guard import GitGuardError, check_full_run, check_resume, repo_state
from .logger import write_json
from .metadata import build_metadata, environment, read_metadata
from .run_dir import create_run_dir, make_run_id, size_report, utc_now


class RunError(RuntimeError):
    """A run cannot be started or resumed."""


@dataclass
class RunResult:
    run_dir: Path
    summary: dict
    warnings: list[str]


def _iso(dt) -> str:
    return dt.isoformat(timespec="seconds")


def _finish(run_dir: Path, metadata: dict, summary: dict) -> RunResult:
    report = size_report(run_dir)
    summary["run_folder"] = report
    write_json(run_dir / "summary.json", summary)
    metadata["status"] = summary["status"]
    metadata["end_utc"] = _iso(utc_now())
    metadata["warnings"] = [*metadata.get("warnings", []), *report["warnings"]]
    write_json(run_dir / "metadata.json", metadata)
    return RunResult(run_dir, summary, metadata["warnings"])


def start_run(
    config_path, *, repo_root, seed=None, smoke=False, device=None, max_iterations=None, command=None
) -> RunResult:
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path, {"experiment": {"seed": seed}} if seed is not None else None)

    if smoke:
        try:
            git_state = repo_state(repo_root)  # recorded for information only
        except GitGuardError:
            git_state = None
        warnings, root = [], repo_root / "outputs" / "smoke"
    else:
        git_state = check_full_run(repo_root, config_path, cfg.experiment.id)
        warnings, root = git_state.pop("warnings"), repo_root / "runs"

    start = utc_now()
    run_id = make_run_id(cfg.experiment.stage, cfg.experiment.seed, start)
    run_dir = create_run_dir(root, cfg.experiment.id, run_id)
    (run_dir / "config.resolved.yaml").write_text(
        f"# Fully resolved config for run {run_dir.name}\n# config_hash: {cfg.config_hash}\n" + cfg.to_yaml(),
        encoding="utf-8",
    )

    trainer = Trainer(cfg, out_dir=run_dir, device=device)
    try:
        source = config_path.relative_to(repo_root).as_posix()
    except ValueError:
        source = str(config_path)
    metadata = build_metadata(
        run_id=run_dir.name, cfg=cfg, scales=Scales.from_config(cfg), git_state=git_state,
        config_source=source, smoke=smoke, command=command, device=trainer.device, start_utc=_iso(start),
    )  # fmt: skip
    metadata["warnings"] = warnings
    write_json(run_dir / "metadata.json", metadata)

    summary = trainer.run(max_iterations=max_iterations)
    return _finish(run_dir, metadata, summary)


def resume_run(run_dir, *, repo_root, device=None, max_iterations=None, command=None) -> RunResult:
    run_dir, repo_root = Path(run_dir).resolve(), Path(repo_root).resolve()
    metadata = read_metadata(run_dir)
    if metadata["status"] == "completed":
        raise RunError(f"{run_dir.name} already completed; nothing to resume")
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise RunError(f"no checkpoint to resume from: {checkpoint}")
    if not metadata["smoke"]:
        check_resume(repo_root, metadata["git"]["commit"])

    cfg = load_config(run_dir / "config.resolved.yaml")
    trainer = Trainer(cfg, out_dir=run_dir, device=device).resume(checkpoint)
    metadata["resume_history"].append({
        "resumed_utc": _iso(utc_now()),
        "from_iteration": trainer.global_iteration,
        "device": environment(trainer.device)["device"],
        "command": command,
    })  # fmt: skip
    metadata["status"] = "running"
    write_json(run_dir / "metadata.json", metadata)

    summary = trainer.run(max_iterations=max_iterations)
    return _finish(run_dir, metadata, summary)
