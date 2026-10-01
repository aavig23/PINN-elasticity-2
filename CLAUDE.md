# Instructions for Claude Code in this repository

## Source of truth
- `PINN_ELASTICITY_SPEC.md` defines the physics. Never change, simplify,
  reinterpret or add physical assumptions. If something is missing or
  ambiguous, ask the user.
- Formulation is **plane stress only** (use λ*, never the 3D λ). There is no
  plane-strain option.
- Training is done only in nondimensional variables; SI only in outputs/figures.
- Implementation decisions are recorded in `docs/decisions.md`. Follow them;
  propose changes explicitly, never silently.

## Permissions (user's standing constraints)
- Do not install packages, make commits or push without explicit approval.
- Do not modify project files unless asked.
- Keep the existing Python 3.13.5 `.venv`; never change the Python version.

## Commands
- Python: `.venv\Scripts\python.exe`
- Tests: `.venv\Scripts\python.exe -m pytest`

## Experiments
- Definitions: `experiments/EXPnnn_<name>/{config.yaml, README.md}`, listed in
  `experiments/INDEX.md`. IDs are assigned locally only.
- Outputs: `runs/EXPnnn/<run_id>/` (committed). Never put runs in `results/`
  or `outputs/` (ignored scratch; smoke runs go to `outputs/smoke/`).
- Never edit anything under `runs/`. Colab commits touch only `runs/`.
- File formats: `docs/run_folder_format.md`.

## Analysing runs (when asked)
1. User has pulled. Find runs via `runs/EXP*/*/summary.json`.
2. Read `summary.json`, `config.resolved.yaml` (diff against parent EXP),
   `metadata.json`, `losses.csv`, and the figures.
3. Report **training behaviour only** (stalling, dominating or oscillating loss
   terms, optimizer phase effects). Do **not** validate against reference/FEM
   solutions or draw accuracy conclusions — validation is the user's (spec §6).
4. Propose concrete config/code changes for the next experiment; write findings
   into the EXP README "Analysis" section only with approval.
