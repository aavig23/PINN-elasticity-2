# Colab setup

The repo is public, so Colab clones it without credentials. A token is needed
only to **push run results** back to GitHub (spec §7.5), which is added in M7.

## Running the notebook
1. Open `notebooks/colab_runner.ipynb` in Colab:
   File → Open notebook → GitHub → `aavig23/PINN-elasticity-2` → `notebooks/colab_runner.ipynb`
   (or `https://colab.research.google.com/github/aavig23/PINN-elasticity-2/blob/main/notebooks/colab_runner.ipynb`).
2. Check Runtime → Change runtime type → **GPU** (the notebook requests one).
3. Optionally set `REF` in cell 2 to a branch, tag or commit SHA (default `main`).
4. Run all cells top to bottom:

| Cell | What it does | Expected result |
|---|---|---|
| 1 | `nvidia-smi` | GPU name and memory |
| 2 | Fresh clone at `REF`, prints the commit | `working tree clean` |
| 3 | `pip install --no-deps -e .`, then `scripts/check_environment.py --install-missing --require-gpu` | same torch version before and after; `ALL CHECKS PASSED` |
| 4 | `python -m pytest -q` | all tests pass |
| 5 | `scripts/smoke_test.py` | both stages `PASS` on `cuda`; `SMOKE TEST PASSED` |

### Why every step is `!python ...`
`pip install -e .` registers the package through a `.pth` file, which the
already-running notebook kernel does not see until the runtime restarts.
Each `!python` command starts a fresh process that does see it, so no restart
is needed.

### Dependency policy (spec §5.2)
- Colab's preinstalled **torch is never installed or upgraded**: it is matched
  to Colab's CUDA drivers. The project is installed with `--no-deps`.
- `check_environment.py --install-missing` installs only packages that are
  **completely missing** (typically none; possibly pytest), at the project's
  lower bounds.
- A package **older** than the project's lower bound is reported as a
  warning, not upgraded, because upgrading Colab's preinstalled packages can
  break others. If a warning appears, record it and decide before relying on
  the run.
- Exact versions of everything are printed by cell 3 (and recorded in every
  run's metadata from M5 on).

### Rules
- Never edit code or configs in Colab. Change them locally, commit, push,
  then re-run the notebook (a fresh clone each time).
- If you save a copy of the notebook back to the repository, clear all
  outputs first (`tests/test_notebook_clean.py` enforces this).

## One-time: create the GitHub token (needed from M7, for pushing results)
1. GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → Generate new token.
2. **Repository access:** *Only select repositories* → `aavig23/PINN-elasticity-2`.
3. **Permissions:** Repository permissions → **Contents: Read and write**.
   Leave everything else at *No access*.
4. **Expiration:** short (e.g. 30 days). Renew when it expires.
5. Copy the token once; GitHub will not show it again.

## One-time: store it in Colab Secrets
1. In Colab, open the key icon (**Secrets**) in the left sidebar.
2. Add a secret named `GITHUB_TOKEN` with the token as its value.
3. Enable **Notebook access** for the runner notebook.

## Token rules
The token must never be hard-coded, typed into a cell, printed, logged,
written to any file, saved in notebook outputs, committed, or left in a git
remote URL or git config. The push script reads it from Colab Secrets at run
time and passes it to git only for the duration of the push.

## Git identity for Colab commits
Commits from Colab use your GitHub **noreply** email
(GitHub → Settings → Emails → "Keep my email addresses private" shows it,
in the form `<id>+<username>@users.noreply.github.com`).

*Experiment runs, Drive checkpoint backup, resume and pushing results are added in M5-M7.*
