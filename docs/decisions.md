# Decision log

Dated record of project decisions. The physics itself is decided in
`PINN_ELASTICITY_SPEC.md`; this file records implementation and workflow
decisions made on top of it. Changes to a decision get a new dated entry;
old entries are never rewritten.

Spec version reviewed: SHA-256 `1E1A4D671E0B395423A43BFA4E9B00CBEED883CF2BC5E660F912FCACA9ADF9E7`.

---

## 2026-10-01 — Project structure and implementation plan approved

### Workflow / infrastructure
| # | Decision |
|---|---|
| A1 | Point 4 of the structure-review message: **pending** (not yet supplied by the user). |
| A2 | GitHub repo `aavig23/PINN-elasticity-2` is **public**. Colab clones without a token; the token is used only to push results. |
| A3 | Mid-run checkpoints on Colab are backed up to **Google Drive** at every checkpoint interval, so a run can resume after a session is lost. |
| A4 | Colab commits use the user's **GitHub noreply email** (user supplies the exact address during Colab setup). |
| A5 | Full (non-smoke) runs **refuse to start** if the working tree is dirty outside `runs/`, or if the config file is not tracked by Git. Smoke runs are exempt and write to `outputs/smoke/`. |
| A6 | Configs: experiment config **extends a base and overrides** only what changes; the fully merged config is saved in each run. Material, geometry and P0 are **locked** to spec values. |
| A7 | Branching: short **feature branch per milestone**, merged to `main` after review. Colab always runs a commit on `main`. Commits and pushes only on the user's instruction. |
| A8 | Package name: `pinn_elasticity`. |
| A9 | Local torch: **CPU build** from the PyTorch CPU index. |
| A10 | Additions to the approved tree: `physics/domain.py`, `notebooks/colab_runner.ipynb`, `CLAUDE.md`, `docs/decisions.md`, `.gitattributes`. |
| — | Colab must never modify source code or experiment configurations; all such changes are made locally, reviewed, committed and pushed first. |
| — | `validation/` directory dropped: validation is done by the user (spec §6) using exported `fields.npz`. |

### Numerical method (resolves spec §8 open item)
| # | Decision |
|---|---|
| B1 | MLP, tanh, Xavier init, inputs affinely mapped to [−1, 1]. Stage 1: 4 hidden × 64. Stage 2: 6 hidden × 128. No Fourier features in baselines. |
| B2 | float32 baseline; each run logs a float32-vs-float64 residual discrepancy check. |
| B3 | **Hard** clamp in both stages (û = x̂·N); **hard** ICs in Stage 2 via x̂·(t̂/T̂)²·N. Traction BCs soft. Soft mode kept available for later experiments. |
| B4 | Adam (lr 1e-3 decaying to 1e-4) then L-BFGS (strong Wolfe). Stage 1: 10 000 + ≤ 2 000. Stage 2: 30 000 + ≤ 5 000. |
| B5 | Fixed loss weights = 1 for baselines; adaptive weighting only when an experiment calls for it. |
| B6 | Uniform sampling (Stage 1: 5 000 interior, 500/500/250 top/bottom/right; Stage 2: 20 000 interior, 4 000/4 000/2 000), resampled every 1 000 Adam iterations, fixed during L-BFGS. No refinement in baselines. |
| B7 | EXP001 runs **3 seeds**. |

### Outputs
| # | Decision |
|---|---|
| C1 | Stage 2 default snapshots: t̂ = 0.5, 1.0, …, 4.0 (8 snapshots; avoids the all-zero t̂ = 0 frame). |
| C2 | Optional ε_yy (spec §1.3) is **not** exported. |
| C3 | Symmetry and global force-balance diagnostics are **not** computed; left to the user's validation. |
| C4 | PDE residual maps are shown in **nondimensional** units (labelled). |
| C5 | Stage 2 runs also write `tip_history.csv` (t̂, t in ms, u in m at (L, H/2)). |

### Environment (M0)
| Item | Value |
|---|---|
| Python (local) | 3.13.5 in `.venv` (unchanged) |
| Installed locally | torch 2.14.1+cpu, numpy 2.5.3, matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.1.1 |
| Repo state | No commits yet, so `main` does not exist as a branch. M0 files are built on the unborn `main`; at the first authorised commit, `.gitignore` + spec go to `main` first and the scaffolding onto `feat/m0-scaffolding`. |
