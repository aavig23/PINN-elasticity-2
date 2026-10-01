# Run folder format

Every experiment run writes one folder (spec §7.3–7.4):

```
runs/EXPnnn/<UTC-timestamp>_<stage>_seed<k>/
```

Example: `runs/EXP001/20261001T153000Z_stage1_seed0/`.
Smoke runs use the same layout under the git-ignored `outputs/smoke/`.

| File | Content |
|---|---|
| `config.resolved.yaml` | Fully merged config actually used, with its SHA-256 `config_hash`. |
| `metadata.json` | Run id, EXP id, git commit hash, branch, dirty flag, sanitised remote URL, start/end time (UTC), platform (local/Colab), device and GPU name, seed, Python/torch/CUDA/cuDNN/numpy/matplotlib/pyyaml versions, T̂ and T in ms, resume history. |
| `losses.csv` | Per log step: iteration, phase, wall time, learning rate, every unweighted loss term, every weight, total, diagnostic metrics. |
| `summary.json` | Status (`running`/`completed`/`failed`/`interrupted`), stage, iterations per phase, training time, final loss terms, term dominance, diagnostics, size warnings. |
| `fields.npz` | Compressed float32, SI units: `u`, `w` (m); `sxx`, `szz`, `sxz` (Pa). Shape `(nz, nx)` for Stage 1, `(nt, nz, nx)` for Stage 2. Grid `x`, `z` (m) and `x_hat`, `z_hat`; times `t` (s) and `t_hat` (Stage 2). |
| `FIELDS.md` | Array names, shapes, axis order, units, grid spacing, scaling constants for `fields.npz`. |
| `tip_history.csv` | Stage 2 only: `t_hat`, `t_ms`, `u_m` at (L, H/2). |
| `figures/` | `loss_curves.png`; `fields_<t>.png` (u, w, σ_xx, σ_zz, σ_xz in SI); `residual_<t>.png` (nondimensional); `tip_history.png` (Stage 2, time in ms). 150 dpi default. |
| `checkpoints/` | `latest.pt` and `final.pt` only: model, optimizer, scheduler, RNG state. |

Stage 1 has no time dimension, so snapshots, tip history and initial-condition
losses are skipped for it.

## Size limits (spec §7.4)
- A run folder should stay under ~20 MB with default settings; larger runs get a
  warning in `summary.json`.
- Any checkpoint over ~50 MB is warned about and excluded from the push.

*This document is completed as the code that writes these files is implemented (milestones M4–M6).*
