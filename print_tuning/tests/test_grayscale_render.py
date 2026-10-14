"""Tests for print_tuning/grayscale_render.py.

Run with `python -m pytest print_tuning/tests`, or without pytest as
`python print_tuning/tests/test_grayscale_render.py`.
"""
import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for running as a script

from print_tuning.grayscale_render import (ConfigError, MappingError, apply_mapping, changed_parameters,
                                           compute_limits, default_mapping, load_config, validate_config,
                                           validate_mapping)


class raises:
    """Like pytest.raises, so the tests also run without pytest."""

    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            raise AssertionError(f"{self.error.__name__} was not raised")
        return issubclass(kind, self.error)


def sample_data():
    rng = np.random.default_rng(0)
    return rng.normal(100.0, 20.0, size=(32, 32))


def test_real_config_loads():
    config = load_config()
    assert config["name"] == "photonic_focus_heatmaps"


def test_round_with_unknown_parameter_is_rejected():
    config = copy.deepcopy(load_config())
    config["rounds"][1]["parameter"] = "contrast"
    with raises(ConfigError):
        validate_config(config)


def test_gray_levels_closer_than_min_gap_are_rejected():
    config = copy.deepcopy(load_config())
    config["parameters"]["gray_levels"]["dark"]["default"] = 0.4
    config["parameters"]["gray_levels"]["light"]["default"] = 0.6
    with raises(ConfigError):
        validate_config(config)


def test_default_mapping_returns_normalized_data():
    config = load_config()
    data = sample_data()
    limits = compute_limits([data], config)
    expected = np.clip((data - limits["lo"]) / (limits["hi"] - limits["lo"]), 0, 1)
    gray = apply_mapping(data, limits, default_mapping(config))
    assert gray.dtype == np.float64 and gray.shape == data.shape
    assert np.max(np.abs(gray - expected)) <= 1e-12


def test_order_is_kept_and_reversed_when_inverted():
    config = load_config()
    values = np.linspace(-1.0, 2.0, 301).reshape(1, -1)
    limits = {"lo": 0.0, "hi": 1.0, "method": "percentile"}
    for gamma in (0.3, 1.0, 3.0):
        mapping = default_mapping(config) | {"gamma": gamma}
        assert np.all(np.diff(apply_mapping(values, limits, mapping)) >= 0)
        inverted = mapping | {"invert": True}
        assert np.all(np.diff(apply_mapping(values, limits, inverted)) <= 0)


def test_steps_give_few_levels_inside_gray_range():
    config = load_config()
    data = sample_data()
    limits = compute_limits([data], config)
    mapping = default_mapping(config) | {"steps": 4, "dark_level": 0.1, "light_level": 0.9}
    validate_mapping(mapping, config)
    gray = apply_mapping(data, limits, mapping)
    assert len(np.unique(gray)) <= 4
    assert gray.min() >= 0.1 and gray.max() <= 0.9


def test_limits_use_all_panels_and_ignore_mapping_and_keep_input():
    config = load_config()
    initial = np.zeros((10, 10))
    final = np.full((10, 10), 50.0)
    initial_copy, final_copy = initial.copy(), final.copy()

    limits = compute_limits([initial, final], config)
    assert limits["lo"] == 0.0 and limits["hi"] == 50.0
    assert compute_limits([initial], config)["flat"] is True

    before = dict(limits)
    for mapping in (default_mapping(config), default_mapping(config) | {"gamma": 3.0, "steps": 8}):
        apply_mapping(initial, limits, mapping)
        apply_mapping(final, limits, mapping)
        assert compute_limits([initial, final], config) == before
    assert np.array_equal(initial, initial_copy) and np.array_equal(final, final_copy)


def test_validate_mapping_rejects_bad_values_and_accepts_smooth():
    config = load_config()
    base = default_mapping(config)
    for bad in ({"gamma": 0.05}, {"steps": 1}, {"dark_level": 0.5}, {"extra": 1}):
        with raises(MappingError):
            validate_mapping(base | bad, config)
    validate_mapping(base | {"steps": 0}, config)


def test_changed_parameters():
    config = load_config()
    base = default_mapping(config)
    assert changed_parameters(base, base | {"gamma": 0.65}) == {"gamma"}
    assert changed_parameters(base, base | {"light_level": 0.9}) == {"gray_levels"}


if __name__ == "__main__":
    tests = [(name, test) for name, test in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
