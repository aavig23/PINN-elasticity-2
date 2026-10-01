# PINN for 2D Plane-Stress Elasticity

A Physics-Informed Neural Network (PyTorch) for a clamped steel plate under
axial tension, in plane stress: first a **static** solve (Stage 1), then the
**elastodynamic** problem (Stage 2).

The physics is defined in **[`PINN_ELASTICITY_SPEC.md`](PINN_ELASTICITY_SPEC.md)**,
the single source of truth for the project.

**Status:** M0 (scaffolding). See [`docs/decisions.md`](docs/decisions.md).

## Repository layout
```
src/pinn_elasticity/   installable package
  physics/             spec equations: material, scaling, domain, constitutive, equations, loading, BCs
  models/              network architectures, hard/soft constraint transforms
  training/            losses, weighting, optimizers, training loop, checkpoints
  postprocess/         grid evaluation, SI conversion, field export, figures
  experiment/          run folders, metadata, logging, git guard
  utils/               autodiff, device, seeding
configs/               shared base and smoke-test configs
experiments/           experiment definitions: EXPnnn_<name>/{config.yaml, README.md}
runs/                  experiment outputs: EXPnnn/<run_id>/   (committed)
scripts/               command-line entry points
notebooks/             Colab launcher
tests/                 pytest suite
docs/                  workflow, Colab setup, run folder format, decision log
```

## Local setup
Uses the existing Python 3.13.5 virtual environment in `.venv`.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest
```

## Colab
Training runs on Google Colab. See [`docs/colab_setup.md`](docs/colab_setup.md)
for the one-time token setup. Notebook instructions are added in M7.

## Documentation
- [`docs/workflow.md`](docs/workflow.md): the local → GitHub → Colab → analysis loop
- [`docs/run_folder_format.md`](docs/run_folder_format.md): what every run saves
- [`docs/colab_setup.md`](docs/colab_setup.md): GitHub token and Colab Secrets
- [`docs/decisions.md`](docs/decisions.md): dated decision log
