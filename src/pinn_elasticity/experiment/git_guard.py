"""Git state checks that tie every full run to exactly one commit (decision A5).

A full (non-smoke) run is refused unless:
  - the repository has no modified or untracked files outside runs/,
  - the config is a tracked file at experiments/EXPnnn_<name>/config.yaml,
  - the config's experiment.id matches its folder (EXPnnn).
Then the recorded commit hash is exactly the code and config that ran.
Smoke runs skip these checks and never write to runs/.
"""

import re
import subprocess
from pathlib import Path

RUNS_DIR = "runs/"
EXPERIMENT_ID = re.compile(r"^EXP\d{3}$")


class GitGuardError(RuntimeError):
    """The repository is not in a state that allows a reproducible full run."""


def git(repo: Path, *args: str, strip: bool = True) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise GitGuardError("git is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise GitGuardError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return result.stdout.strip() if strip else result.stdout


def sanitize_url(url: str) -> str:
    """Remove any credentials embedded in a remote URL (https://user:token@host/...)."""
    return re.sub(r"//[^/@]+@", "//", url)


def changed_paths(repo: Path) -> list[str]:
    """Modified, staged or untracked paths (ignored files excluded), relative to the repo.

    Uses NUL-separated porcelain output: entries are "XY path", never quoted, and
    must not be whitespace-stripped (X is a space for unstaged changes). A rename
    or copy entry is followed by a separate entry holding the original path.
    """
    entries = git(repo, "status", "--porcelain", "-z", "--untracked-files=all", strip=False).split("\0")
    paths, i = [], 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if status[0] in "RC":  # skip the original path of a rename/copy
            i += 1
    return paths


def repo_state(repo: Path) -> dict:
    """Commit, branch, dirtiness (outside runs/) and remote status of ``repo``."""
    repo = Path(repo)
    if git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise GitGuardError(f"{repo} is not a git working tree")
    code_changes = [p for p in changed_paths(repo) if not p.startswith(RUNS_DIR)]
    try:
        remote_url = sanitize_url(git(repo, "remote", "get-url", "origin"))
    except GitGuardError:
        remote_url = None
    on_remote = bool(git(repo, "branch", "-r", "--contains", "HEAD")) if remote_url else False
    return {
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(code_changes),
        "dirty_paths": code_changes[:50],
        "remote_url": remote_url,
        "commit_on_remote": on_remote,
    }


def check_full_run(repo: Path, config_path: Path, experiment_id: str) -> dict:
    """Raise GitGuardError unless a full run from ``config_path`` is reproducible.

    Returns the repo state (see repo_state), including a "warnings" list.
    """
    repo, config_path = Path(repo).resolve(), Path(config_path).resolve()
    state = repo_state(repo)

    if state["dirty"]:
        listed = "\n  ".join(state["dirty_paths"])
        raise GitGuardError(
            "working tree has changes outside runs/; commit (and push) them before a full run, "
            f"or use --smoke:\n  {listed}"
        )

    try:
        rel = config_path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise GitGuardError(f"config {config_path} is outside the repository") from exc
    parts = rel.split("/")
    if len(parts) != 3 or parts[0] != "experiments" or parts[2] != "config.yaml":
        raise GitGuardError(f"full runs need a config at experiments/EXPnnn_<name>/config.yaml, got {rel}")
    if not EXPERIMENT_ID.match(experiment_id):
        raise GitGuardError(f"experiment.id must look like EXP001, got {experiment_id!r}")
    if not parts[1].startswith(experiment_id + "_"):
        raise GitGuardError(f"experiment.id {experiment_id} does not match its folder {parts[1]}")
    try:
        git(repo, "ls-files", "--error-unmatch", rel)
    except GitGuardError as exc:
        raise GitGuardError(f"config {rel} is not tracked by git; commit it first") from exc

    state["warnings"] = []
    if not state["commit_on_remote"]:
        state["warnings"].append(
            "HEAD is not on any remote branch; push it so the recorded commit can be found on GitHub"
        )
    return state


def check_resume(repo: Path, recorded_commit: str) -> dict:
    """Raise GitGuardError unless the checkout is clean and at the run's recorded commit."""
    state = repo_state(repo)
    if state["dirty"]:
        raise GitGuardError("working tree has changes outside runs/; a resumed run must use the recorded code")
    if state["commit"] != recorded_commit:
        raise GitGuardError(
            f"HEAD is {state['commit'][:12]} but the run was started at {recorded_commit[:12]}; "
            "check out the recorded commit to resume"
        )
    state["warnings"] = []
    return state
