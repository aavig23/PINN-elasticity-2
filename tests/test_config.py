import pytest

from pinn_elasticity.config import ConfigError, load_config


def test_round_trip_through_yaml(make_config, tmp_path):
    cfg = make_config(1)
    path = tmp_path / "resolved.yaml"
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    again = load_config(path)
    assert again == cfg
    assert again.config_hash == cfg.config_hash


def test_overrides_deep_merge(make_config):
    cfg = make_config(1, {"loss": {"weights": {"pde": 2.0}}})
    assert cfg.loss.weights == {"pde": 2.0, "free": 1.0, "load": 1.0, "clamp": 1.0, "ic": 1.0}


def test_hash_changes_with_config(make_config):
    assert make_config(1).config_hash != make_config(1, {"experiment": {"seed": 1}}).config_hash


def test_time_parameters_are_configurable(make_config):
    cfg = make_config(2, {"time": {"T_hat": 6.0, "t_r_hat": 2.0}})
    assert (cfg.time.T_hat, cfg.time.t_r_hat) == (6.0, 2.0)
    assert cfg.snapshot_times()[-1] == 6.0


@pytest.mark.parametrize("key, value", [("E", 210e9), ("nu", 0.25), ("H", 0.1), ("P0", 2000.0)])
def test_physics_is_locked_to_spec(make_config, key, value):
    with pytest.raises(ConfigError, match="locked"):
        make_config(1, {"physics": {key: value}})


def test_unknown_key_rejected(make_config):
    with pytest.raises(ConfigError, match="unknown"):
        make_config(1, {"model": {"depth": 3}})


@pytest.mark.parametrize(
    "overrides",
    [
        {"constraints": {"mode": "plane_strain"}},
        {"scaling": {"mode": "custom"}},
        {"model": {"activation": "relu"}},
        {"experiment": {"stage": 3}},
        {"precision": {"dtype": "float16"}},
        {"output": {"snapshot_times": [0.5, 9.0]}},
    ],
)
def test_invalid_values_rejected(make_config, overrides):
    with pytest.raises(ConfigError):
        make_config(2, overrides)


def test_soft_stage2_requires_initial_points(make_config):
    with pytest.raises(ConfigError, match="n_initial"):
        make_config(2, {"constraints": {"mode": "soft"}, "sampling": {"n_initial": 0}})


def test_missing_key_rejected(tmp_path, make_config):
    raw = make_config(1).to_dict()
    del raw["model"]["width"]
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_config(path)


def test_unsigned_exponent_gives_helpful_error(tmp_path, make_config):
    import yaml

    raw = make_config(1).to_dict()
    text = yaml.safe_dump(raw).replace("E: 200000000000.0", "E: 200.0e9")
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="exponent"):
        load_config(path)


def test_circular_extends_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")
