"""metadata.json: what ran, where, with which code and environment (spec §5.2, §7.4)."""

import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path

import torch

from ..utils.device import describe_device

VERSIONED_PACKAGES = ("torch", "numpy", "matplotlib", "pyyaml")


def environment(device: torch.device) -> dict:
    cuda = device.type == "cuda"
    versions = {"python": sys.version.split()[0]}
    for dist in VERSIONED_PACKAGES:
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    versions["cuda"] = torch.version.cuda
    versions["cudnn"] = torch.backends.cudnn.version() if cuda else None
    return {
        "platform": {
            "system": platform.platform(),
            "colab": "COLAB_RELEASE_TAG" in os.environ,
        },
        "device": {"type": device.type, "name": describe_device(device)},
        "versions": versions,
    }


def build_metadata(*, run_id, cfg, scales, git_state, config_source, smoke, command, device, start_utc) -> dict:
    stage = cfg.experiment.stage
    return {
        "run_id": run_id,
        "experiment_id": cfg.experiment.id,
        "experiment_name": cfg.experiment.name,
        "parent": cfg.experiment.parent,
        "stage": stage,
        "seed": cfg.experiment.seed,
        "smoke": smoke,
        "status": "running",
        "config_source": config_source,
        "config_hash": cfg.config_hash,
        "command": command,
        "git": git_state,
        "start_utc": start_utc,
        "end_utc": None,
        "time_window": (
            {"T_hat": cfg.time.T_hat, "T_ms": scales.time_ms(cfg.time.T_hat)} if stage == 2 else None
        ),
        **environment(device),
        "resume_history": [],
    }


def read_metadata(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
