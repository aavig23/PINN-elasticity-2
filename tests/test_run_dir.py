import re
from datetime import datetime, timezone

import pinn_elasticity.experiment.run_dir as run_dir_module
from pinn_elasticity.experiment.run_dir import create_run_dir, make_run_id, size_report


def test_run_id_format():
    now = datetime(2026, 10, 1, 15, 30, 0, tzinfo=timezone.utc)
    assert make_run_id(1, 0, now) == "20261001T153000Z_stage1_seed0"
    assert re.fullmatch(r"\d{8}T\d{6}Z_stage2_seed7", make_run_id(2, 7))


def test_run_folders_are_never_reused(tmp_path):
    a = create_run_dir(tmp_path, "EXP001", "20261001T153000Z_stage1_seed0")
    b = create_run_dir(tmp_path, "EXP001", "20261001T153000Z_stage1_seed0")
    assert a == tmp_path / "EXP001" / "20261001T153000Z_stage1_seed0"
    assert b.name == "20261001T153000Z_stage1_seed0_2"
    assert a.is_dir() and b.is_dir()


def test_size_report_small_folder_has_no_warnings(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "final.pt").write_bytes(b"x" * 1000)
    report = size_report(tmp_path)
    assert report == {"size_mb": 0.001, "oversized_checkpoints": [], "warnings": []}


def test_size_report_flags_large_folder_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dir_module, "FOLDER_LIMIT_MB", 0.001)
    monkeypatch.setattr(run_dir_module, "CHECKPOINT_LIMIT_MB", 0.001)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "latest.pt").write_bytes(b"x" * 5000)
    report = size_report(tmp_path)
    assert report["oversized_checkpoints"] == ["checkpoints/latest.pt"]
    assert len(report["warnings"]) == 2
