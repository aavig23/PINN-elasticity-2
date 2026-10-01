import json
import math

import pytest
import torch

import pinn_elasticity.training.trainer as trainer_module
from pinn_elasticity.experiment.logger import read_losses
from pinn_elasticity.training.trainer import Trainer

FLOAT64 = {"precision": {"dtype": "float64"}}


def test_stage1_smoke_run_completes_and_reduces_loss(make_config, tmp_path):
    summary = Trainer(make_config(1, FLOAT64), out_dir=tmp_path, device="cpu").run()
    assert summary["status"] == "completed"
    assert summary["iterations"] == {"adam": 40, "lbfgs": 10, "total": 50}

    rows = read_losses(tmp_path / "losses.csv")
    assert rows[0]["iteration"] == "0" and rows[-1]["phase"] == "final"
    assert float(rows[-1]["total"]) < float(rows[0]["total"])
    assert set(rows[0]) >= {"pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz", "w_pde", "pde_excl_corners"}

    assert (tmp_path / "checkpoints" / "latest.pt").is_file()
    assert (tmp_path / "checkpoints" / "final.pt").is_file()
    assert not list(tmp_path.rglob("*.tmp"))
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["final"]["total"] == summary["final"]["total"]
    assert sum(summary["final"]["dominance"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_stage2_smoke_run_completes(make_config, mode):
    cfg = make_config(2, {"constraints": {"mode": mode}, "optimizer": {"adam_iterations": 10, "lbfgs_iterations": 2}})
    summary = Trainer(cfg, device="cpu").run()
    assert summary["status"] == "completed"
    assert math.isfinite(summary["final"]["total"])
    assert summary["T_ms"] == pytest.approx(0.79, rel=5e-3)
    if mode == "soft":
        assert {"ic_u", "ic_w", "ic_ut", "ic_wt", "clamp_u", "clamp_w"} <= set(summary["final"]["terms"])


@pytest.mark.parametrize("stop_at", [8, 15])  # inside the Adam phase and inside the L-BFGS phase
def test_resume_reproduces_an_uninterrupted_run(make_config, tmp_path, stop_at):
    cfg = make_config(1, {
        **FLOAT64,
        "sampling": {"resample_every": 5},
        "optimizer": {"adam_iterations": 12, "lbfgs_iterations": 6},
        "training": {"log_every": 1, "lbfgs_log_every": 1, "checkpoint_every": 100},
    })  # fmt: skip

    straight = Trainer(cfg, out_dir=tmp_path / "a", device="cpu")
    straight.run()

    first = Trainer(cfg, out_dir=tmp_path / "b", device="cpu")
    assert first.run(max_iterations=stop_at)["status"] == "interrupted"
    resumed = Trainer(cfg, out_dir=tmp_path / "b", device="cpu").resume(tmp_path / "b" / "checkpoints" / "latest.pt")
    summary = resumed.run()
    assert summary["status"] == "completed" and summary["resume_count"] == 1

    for p, q in zip(straight.model.parameters(), resumed.model.parameters()):
        assert torch.equal(p, q)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in rows]  # noqa: E731
    assert strip(read_losses(tmp_path / "a" / "losses.csv")) == strip(read_losses(tmp_path / "b" / "losses.csv"))


def test_resume_rejects_a_different_config(make_config, tmp_path):
    Trainer(make_config(1), out_dir=tmp_path, device="cpu").run(max_iterations=3)
    other = Trainer(make_config(1, {"experiment": {"seed": 7}}), device="cpu")
    with pytest.raises(ValueError, match="config hash"):
        other.resume(tmp_path / "checkpoints" / "latest.pt")


def test_non_finite_loss_fails_cleanly(make_config, tmp_path, monkeypatch):
    real = trainer_module.weighted_total
    monkeypatch.setattr(trainer_module, "weighted_total", lambda terms, weights: real(terms, weights) * math.nan)
    summary = Trainer(make_config(1), out_dir=tmp_path, device="cpu").run()
    assert summary["status"] == "failed" and "non-finite" in summary["message"]
    assert not (tmp_path / "checkpoints" / "final.pt").exists()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"
