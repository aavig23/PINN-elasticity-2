import pytest

from conftest import TEST_EXPERIMENT, run_git
from pinn_elasticity.experiment.git_guard import (
    GitGuardError,
    changed_paths,
    check_full_run,
    check_resume,
    repo_state,
    sanitize_url,
)


def _config(repo):
    return repo / "experiments" / TEST_EXPERIMENT / "config.yaml"


def test_sanitize_url_removes_credentials():
    assert sanitize_url("https://x-access-token:SECRET@github.com/a/b.git") == "https://github.com/a/b.git"
    assert sanitize_url("https://github.com/a/b.git") == "https://github.com/a/b.git"


def test_clean_repo_allows_full_run(git_repo):
    state = check_full_run(git_repo, _config(git_repo), "EXP999")
    assert state["commit"] == run_git(git_repo, "rev-parse", "HEAD")
    assert state["dirty"] is False and state["branch"] == "main"
    assert any("not on any remote" in w for w in state["warnings"])  # no remote in the test repo


def test_pushed_commit_gives_no_warning(git_repo, tmp_path):
    remote = tmp_path / "remote.git"
    run_git(tmp_path, "init", "-q", "--bare", str(remote))
    run_git(git_repo, "remote", "add", "origin", str(remote))
    run_git(git_repo, "push", "-q", "origin", "main")
    run_git(git_repo, "fetch", "-q", "origin")
    state = check_full_run(git_repo, _config(git_repo), "EXP999")
    assert state["commit_on_remote"] is True and state["warnings"] == []


def test_modified_tracked_file_refused(git_repo):
    (git_repo / "configs" / "base" / "physics.yaml").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(GitGuardError, match="changes outside runs/"):
        check_full_run(git_repo, _config(git_repo), "EXP999")


def test_untracked_file_refused(git_repo):
    (git_repo / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(GitGuardError, match="notes.txt"):
        check_full_run(git_repo, _config(git_repo), "EXP999")


def test_changes_under_runs_and_ignored_scratch_are_allowed(git_repo):
    (git_repo / "runs" / "EXP999" / "r1").mkdir(parents=True)
    (git_repo / "runs" / "EXP999" / "r1" / "losses.csv").write_text("x", encoding="utf-8")
    (git_repo / "outputs" / "smoke").mkdir(parents=True)
    (git_repo / "outputs" / "smoke" / "a.txt").write_text("x", encoding="utf-8")
    assert repo_state(git_repo)["dirty"] is False


def test_changed_paths_are_exact_for_unstaged_and_renamed_files(git_repo):
    # Regression: whitespace-stripping the porcelain output dropped the first
    # character of an unstaged path (" M scripts/run.py" -> "cripts/run.py").
    with (git_repo / "scripts" / "run.py").open("a", encoding="utf-8") as fh:
        fh.write("# edit\n")
    assert changed_paths(git_repo) == ["scripts/run.py"]
    run_git(git_repo, "checkout", "--", "scripts/run.py")
    run_git(git_repo, "mv", ".gitignore", "renamed.gitignore")
    assert changed_paths(git_repo) == ["renamed.gitignore"]


def test_modified_tracked_file_under_runs_is_allowed(git_repo):
    # Regression: a mangled path ("uns/...") was treated as a code change.
    (git_repo / "runs" / "EXP999").mkdir(parents=True)
    (git_repo / "runs" / "EXP999" / "summary.json").write_text("{}", encoding="utf-8")
    run_git(git_repo, "add", "runs")
    run_git(git_repo, "commit", "-q", "-m", "run")
    (git_repo / "runs" / "EXP999" / "summary.json").write_text('{"x": 1}', encoding="utf-8")
    assert changed_paths(git_repo) == ["runs/EXP999/summary.json"]
    assert repo_state(git_repo)["dirty"] is False


def test_config_outside_experiments_refused(git_repo):
    with pytest.raises(GitGuardError, match="experiments/EXPnnn_<name>/config.yaml"):
        check_full_run(git_repo, git_repo / "configs" / "smoke" / "stage1_smoke.yaml", "SMOKE")


def test_untracked_ignored_config_refused(git_repo):
    exp = git_repo / "experiments" / "EXP998_ignored"
    exp.mkdir()
    (exp / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    (git_repo / ".git" / "info" / "exclude").write_text("experiments/EXP998_ignored/\n", encoding="utf-8")
    with pytest.raises(GitGuardError, match="not tracked"):
        check_full_run(git_repo, exp / "config.yaml", "EXP998")


@pytest.mark.parametrize("experiment_id, message", [("EXP001", "does not match"), ("BASE", "look like EXP001")])
def test_experiment_id_must_match_folder(git_repo, experiment_id, message):
    with pytest.raises(GitGuardError, match=message):
        check_full_run(git_repo, _config(git_repo), experiment_id)


def test_resume_requires_the_recorded_commit(git_repo):
    first = run_git(git_repo, "rev-parse", "HEAD")
    check_resume(git_repo, first)
    (git_repo / "README.md").write_text("new\n", encoding="utf-8")
    run_git(git_repo, "add", "README.md")
    run_git(git_repo, "commit", "-q", "-m", "change")
    with pytest.raises(GitGuardError, match="check out the recorded commit"):
        check_resume(git_repo, first)
