"""Checkpoint files (spec §7.2, §7.4): only latest.pt and final.pt per run.

Saving writes to ``<name>.tmp`` and then renames, so a disconnect mid-save
never leaves a corrupt checkpoint (the .tmp files are git-ignored).
"""

import os
from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location) -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)
