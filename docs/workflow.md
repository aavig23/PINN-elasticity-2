# Development and experiment workflow

```
Local VS Code + Claude Code      write code/configs, run tests and smoke runs
        ↓  review, commit, push
GitHub (main)                    single source of truth
        ↓  clone / pull
Google Colab                     run the experiment on GPU (never edits code/configs)
        ↓  push run folder only
GitHub runs/EXPnnn/<run_id>/
        ↓  git pull
Claude Code analysis             training behaviour only (no accuracy claims)
        ↓
Next experiment (new EXP folder) ↺
```

## What runs where
| Local (CPU) | Colab (GPU) |
|---|---|
| Full `pytest` suite; smoke runs of both stages; quick checks | Every experiment run; resumes after disconnects |

## Experiments vs runs
- **Experiment** = a question or change, defined in `experiments/EXPnnn_<name>/`
  (`config.yaml` + `README.md`). IDs are three digits, assigned locally only.
- **Run** = one execution of an experiment, stored in `runs/EXPnnn/<run_id>/`.
  One experiment can have several runs (e.g. seeds).
- `experiments/INDEX.md` lists every experiment. Edited locally only.

## Identifying exactly what ran
A run is identified by **run id + EXP id + git commit hash + config hash**, all
in its `metadata.json`. Full runs refuse to start on a modified working tree,
so the recorded commit is exactly the code that ran.

## Rules
- Colab never modifies source code or experiment configs.
- Local work never edits anything under `runs/`; Colab commits touch only
  `runs/`. The two never conflict.
- Commits and pushes happen only when the user asks.

*Commands for each step are added as they are implemented (M5–M7).*
