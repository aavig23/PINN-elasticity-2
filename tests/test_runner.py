"""End-to-end run folders in a throwaway git repo (tiny config, CPU)."""

import json
import subprocess
import sys

import pytest

from conftest import TEST_EXPERIMENT, run_git
from pinn_elasticity.config import load_config
from pinn_elasticity.experiment.git_guard import GitGuardError
from pinn_elasticity.experiment.metadata import read_metadata
from pinn_elasticity.experiment.runner import RunError, resume_run, start_run

RUN_FILES = {"config.resolved.yaml", "metadata.json", "losses.csv", "summary.json",
             "checkpoints/latest.pt", "checkpoints/final.pt"}  # fmt: skip


def _config(repo):
    return repo / "experiments" / TEST_EXPERIMENT / "config.yaml"


def _files(run_dir):
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}


def test_full_run_creates_complete_run_folder(git_repo):
    result = start_run(_config(git_repo), repo_root=git_repo, device="cpu", command=["test"])
    run_dir = result.run_dir
    assert run_dir.parent == git_repo / "runs" / "EXP999"
    assert run_dir.name.endswith("_stage1_seed0")
    assert _files(run_dir) == RUN_FILES

    meta = read_metadata(run_dir)
    assert meta["status"] == "completed" and meta["end_utc"] is not None
    assert meta["git"]["commit"] == run_git(git_repo, "rev-parse", "HEAD")
    assert meta["git"]["dirty"] is False
    assert meta["experiment_id"] == "EXP999" and meta["seed"] == 0 and meta["smoke"] is False
    assert meta["config_source"] == f"experiments/{TEST_EXPERIMENT}/config.yaml"
    assert meta["versions"]["torch"] and meta["versions"]["python"]
    assert meta["device"]["type"] == "cpu"
    assert meta["time_window"] is None  # Stage 1

    resolved = load_config(run_dir / "config.resolved.yaml")
    assert resolved.config_hash == meta["config_hash"] == result.summary["config_hash"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_folder"]["size_mb"] > 0 and summary["run_folder"]["warnings"] == []


def test_seed_override_is_recorded(git_repo):
    result = start_run(_config(git_repo), repo_root=git_repo, seed=2, device="cpu")
    assert result.run_dir.name.endswith("_seed2")
    assert read_metadata(result.run_dir)["seed"] == 2
    assert load_config(result.run_dir / "config.resolved.yaml").experiment.seed == 2


def test_dirty_repo_refused_and_no_run_folder_created(git_repo):
    with (git_repo / "scripts" / "run.py").open("a", encoding="utf-8") as fh:
        fh.write("# uncommitted edit\n")
    with pytest.raises(GitGuardError, match="scripts/run.py"):
        start_run(_config(git_repo), repo_root=git_repo, device="cpu")
    assert not (git_repo / "runs").exists()


def test_smoke_run_skips_git_checks_and_writes_to_outputs(git_repo):
    (git_repo / "notes.txt").write_text("uncommitted", encoding="utf-8")
    result = start_run(git_repo / "configs" / "smoke" / "stage1_smoke.yaml", repo_root=git_repo,
                       smoke=True, device="cpu", max_iterations=3)  # fmt: skip
    assert result.run_dir.parent == git_repo / "outputs" / "smoke" / "SMOKE"
    meta = read_metadata(result.run_dir)
    assert meta["smoke"] is True and meta["git"]["dirty"] is True
    assert not (git_repo / "runs").exists()


def test_interrupted_run_resumes_to_completion(git_repo):
    first = start_run(_config(git_repo), repo_root=git_repo, device="cpu", max_iterations=4)
    assert first.summary["status"] == "interrupted"
    assert read_metadata(first.run_dir)["status"] == "interrupted"

    resumed = resume_run(first.run_dir, repo_root=git_repo, device="cpu", command=["resume"])
    assert resumed.summary["status"] == "completed"
    meta = read_metadata(first.run_dir)
    assert meta["status"] == "completed"
    assert [h["from_iteration"] for h in meta["resume_history"]] == [4]

    with pytest.raises(RunError, match="already completed"):
        resume_run(first.run_dir, repo_root=git_repo, device="cpu")


def test_resume_refused_after_code_changes(git_repo):
    first = start_run(_config(git_repo), repo_root=git_repo, device="cpu", max_iterations=4)
    (git_repo / "README.md").write_text("change\n", encoding="utf-8")
    run_git(git_repo, "add", "README.md")
    run_git(git_repo, "commit", "-q", "-m", "change")
    with pytest.raises(GitGuardError, match="recorded commit"):
        resume_run(first.run_dir, repo_root=git_repo, device="cpu")


def _cli(repo, *args):
    return subprocess.run([sys.executable, "scripts/run.py", *args], cwd=repo,
                          capture_output=True, text=True, timeout=600)  # fmt: skip


def test_cli_full_run_and_refusal(git_repo):
    ok = _cli(git_repo, "--config", f"experiments/{TEST_EXPERIMENT}/config.yaml", "--device", "cpu")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "status      completed" in ok.stdout
    assert "not on any remote branch" in ok.stdout  # warning shown for an unpushed commit

    (git_repo / "notes.txt").write_text("x", encoding="utf-8")
    refused = _cli(git_repo, "--config", f"experiments/{TEST_EXPERIMENT}/config.yaml", "--device", "cpu")
    assert refused.returncode == 1 and "REFUSED" in refused.stderr


def test_cli_interrupt_exit_code(git_repo):
    result = _cli(git_repo, "--config", "configs/smoke/stage1_smoke.yaml", "--smoke",
                  "--device", "cpu", "--max-iterations", "3")  # fmt: skip
    assert result.returncode == 2, result.stdout + result.stderr
