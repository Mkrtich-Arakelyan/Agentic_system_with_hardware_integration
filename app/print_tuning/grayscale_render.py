"""Turn data into printed gray levels with the tuned settings from tuning_config.json.

A mapping only changes how data looks on paper, never the data itself. Grays run
from 0 (black) to 1 (white). A mapping is a dict with exactly these keys:
  gamma (float), dark_level (float), light_level (float), steps (int), invert (bool).
dark_level and light_level come from the config's gray_levels parameter. Only gamma is
tuned; gray_levels and steps are fixed at their config defaults (dark 0, light 1, smooth).
"""
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "tuning_config.json"

TOP_LEVEL_KEYS = ("schema_version", "name", "description", "page", "panels", "limits",
                  "parameters", "rounds", "variation_rules", "references", "scan")

# mapping key -> config parameter it belongs to
MAPPING_PARAMETERS = {"gamma": "gamma", "dark_level": "gray_levels", "light_level": "gray_levels",
                      "steps": "steps", "invert": "invert"}


class ConfigError(ValueError):
    """The tuning config is missing something or has values that do not fit together."""


class MappingError(ValueError):
    """A mapping has missing or extra keys, wrong types or values out of range."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(config: dict, section: str, key: str) -> int:
    value = config[section].get(key)
    if not _is_int(value) or value < 1:
        raise ConfigError(f"{section}.{key} must be a positive integer, got {value!r}")
    return value


def _check_bounds(where: str, spec: dict, special_values: dict | None = None) -> None:
    """Check that spec has numeric min < max and a default inside, or listed in special_values."""
    low, high, default = spec.get("min"), spec.get("max"), spec.get("default")
    if not (_is_number(low) and _is_number(high) and _is_number(default)):
        raise ConfigError(f"{where} needs numeric min, max and default, got {low!r}, {high!r}, {default!r}")
    if low >= high:
        raise ConfigError(f"{where}: min {low} must be smaller than max {high}")
    if not low <= default <= high and str(default) not in (special_values or {}):
        raise ConfigError(f"{where}: default {default} is outside [{low}, {high}] "
                          "and not in special_values")


def load_config(path: str | Path | None = None) -> dict:
    """Read the tuning config (default: config/tuning_config.json in the project root), validate and return it."""
    with open(path or CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Raise ConfigError with a clear message if the config is incomplete or inconsistent."""
    if not isinstance(config, dict):
        raise ConfigError("the config must be a JSON object")
    missing = [key for key in TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ConfigError(f"missing top-level keys: {', '.join(missing)}")

    _positive_int(config, "page", "items_per_page")
    _positive_int(config, "page", "variations_per_item")
    low = _positive_int(config, "panels", "min_per_item")
    high = _positive_int(config, "panels", "max_per_item")
    if low > high:
        raise ConfigError(f"panels.min_per_item {low} is larger than panels.max_per_item {high}")

    limits = config["limits"]
    if limits.get("method") != "percentile":
        raise ConfigError(f"limits.method must be 'percentile', got {limits.get('method')!r}")
    low, high = limits.get("low_percentile"), limits.get("high_percentile")
    if not (_is_number(low) and _is_number(high) and 0 <= low < high <= 100):
        raise ConfigError(f"limits need 0 <= low_percentile < high_percentile <= 100, got {low!r}, {high!r}")

    parameters = config["parameters"]
    if not isinstance(parameters, dict) or not parameters:
        raise ConfigError("parameters must be a non-empty object")
    for name, spec in parameters.items():
        where = f"parameters.{name}"
        if spec.get("scope") not in ("per_item", "shared"):
            raise ConfigError(f"{where}.scope must be 'per_item' or 'shared', got {spec.get('scope')!r}")
        if not isinstance(spec.get("tuned"), bool):
            raise ConfigError(f"{where}.tuned must be true or false")
        kind = spec.get("type")
        if kind == "float":
            _check_bounds(where, spec)
        elif kind == "int":
            if not all(_is_int(spec.get(key)) for key in ("min", "max", "default")):
                raise ConfigError(f"{where} needs integer min, max and default")
            _check_bounds(where, spec, spec.get("special_values"))
        elif kind == "range":
            for end in ("dark", "light"):
                if not isinstance(spec.get(end), dict):
                    raise ConfigError(f"{where}.{end} must be an object with min, max and default")
                _check_bounds(f"{where}.{end}", spec[end])
            gap = spec.get("min_gap")
            if not _is_number(gap) or gap < 0:
                raise ConfigError(f"{where}.min_gap must be a number >= 0, got {gap!r}")
            if spec["light"]["default"] - spec["dark"]["default"] < gap:
                raise ConfigError(f"{where}: default light minus dark is smaller than min_gap {gap}")
        elif kind == "bool":
            if not isinstance(spec.get("default"), bool):
                raise ConfigError(f"{where}.default must be true or false")
        else:
            raise ConfigError(f"{where}.type must be float, int, range or bool, got {kind!r}")

    rounds = config["rounds"]
    if not isinstance(rounds, list) or not rounds:
        raise ConfigError("rounds must be a non-empty list")
    for number, entry in enumerate(rounds, start=1):
        if entry.get("round_id") != number:
            raise ConfigError(f"round_ids must be 1, 2, 3, ... with no gaps; "
                              f"entry {number} has round_id {entry.get('round_id')!r}")
        name = entry.get("parameter")
        if name not in parameters:
            raise ConfigError(f"round {number} names unknown parameter {name!r}")
        if parameters[name]["tuned"] is not True:
            raise ConfigError(f"round {number} names parameter {name!r}, which is not tuned")


def default_mapping(config: dict) -> dict:
    """Return the mapping made of the config's default values."""
    parameters = config["parameters"]
    return {
        "gamma": float(parameters["gamma"]["default"]),
        "dark_level": float(parameters["gray_levels"]["dark"]["default"]),
        "light_level": float(parameters["gray_levels"]["light"]["default"]),
        "steps": int(parameters["steps"]["default"]),
        "invert": bool(parameters["invert"]["default"]),
    }


def validate_mapping(mapping: dict, config: dict) -> None:
    """Raise MappingError if the mapping has wrong keys, types or values for this config."""
    if not isinstance(mapping, dict):
        raise MappingError("a mapping must be a dict")
    missing = set(MAPPING_PARAMETERS) - set(mapping)
    extra = set(mapping) - set(MAPPING_PARAMETERS)
    if missing or extra:
        raise MappingError(f"mapping keys are wrong: missing {sorted(missing)}, extra {sorted(extra)}")
    for key in ("gamma", "dark_level", "light_level"):
        if not _is_number(mapping[key]):
            raise MappingError(f"{key} must be a number, got {mapping[key]!r}")
    if not _is_int(mapping["steps"]):
        raise MappingError(f"steps must be an integer, got {mapping['steps']!r}")
    if not isinstance(mapping["invert"], bool):
        raise MappingError(f"invert must be true or false, got {mapping['invert']!r}")

    parameters = config["parameters"]
    gray = parameters["gray_levels"]
    for key, spec in (("gamma", parameters["gamma"]), ("dark_level", gray["dark"]),
                      ("light_level", gray["light"])):
        if not spec["min"] <= mapping[key] <= spec["max"]:
            raise MappingError(f"{key} {mapping[key]} is outside [{spec['min']}, {spec['max']}]")
    if mapping["light_level"] - mapping["dark_level"] < gray["min_gap"]:
        raise MappingError(f"light_level - dark_level is {mapping['light_level'] - mapping['dark_level']:g}, "
                           f"smaller than min_gap {gray['min_gap']}")
    steps = parameters["steps"]
    if str(mapping["steps"]) not in steps.get("special_values", {}) and \
            not steps["min"] <= mapping["steps"] <= steps["max"]:
        raise MappingError(f"steps {mapping['steps']} is not in special_values "
                           f"{sorted(steps.get('special_values', {}))} or in [{steps['min']}, {steps['max']}]")


def compute_limits(arrays: list, config: dict) -> dict:
    """Return {"lo", "hi", "method"}: the config's percentiles over all of an item's panels together.

    Flat data (hi <= lo) gets hi = lo + 1 and "flat": True.
    """
    if not arrays:
        raise ValueError("compute_limits needs at least one array")
    values = []
    for array in arrays:
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"each panel must be a 2D array, got {array.ndim} dimensions")
        if array.size == 0:
            raise ValueError("a panel is empty")
        if not np.all(np.isfinite(array)):
            raise ValueError("a panel has NaN or infinite values")
        values.append(array.ravel())
    values = np.concatenate(values)
    settings = config["limits"]
    lo, hi = np.percentile(values, [settings["low_percentile"], settings["high_percentile"]])
    limits = {"lo": float(lo), "hi": float(hi), "method": settings["method"]}
    if limits["hi"] <= limits["lo"]:
        limits["hi"] = limits["lo"] + 1.0
        limits["flat"] = True
    return limits


def apply_mapping(data, limits: dict, mapping: dict) -> np.ndarray:
    """Return the printed grays (float64, same shape as data, in [dark_level, light_level]).

    The input is never modified.
    """
    data = np.asarray(data, dtype=np.float64)
    lo, hi = limits["lo"], limits["hi"]
    y = np.clip((data - lo) / (hi - lo), 0.0, 1.0) ** mapping["gamma"]
    if mapping["invert"]:
        y = 1.0 - y
    steps = mapping["steps"]
    if steps >= 2:
        y = np.round(y * (steps - 1)) / (steps - 1)
    dark, light = mapping["dark_level"], mapping["light_level"]
    return np.clip(dark + (light - dark) * y, dark, light)


def scale_bar(limits: dict, mapping: dict, n: int = 256) -> dict:
    """Return {"values", "grays", "ticks"} for a scale bar from lo to hi; ticks are lo, middle, hi."""
    lo, hi = limits["lo"], limits["hi"]
    values = np.linspace(lo, hi, n)
    ticks = [{"value": value, "label": f"{value:.3g}"} for value in (lo, (lo + hi) / 2, hi)]
    return {"values": values, "grays": apply_mapping(values, limits, mapping), "ticks": ticks}


def mapping_label(mapping: dict) -> str:
    """Short tile label, e.g. "γ 0.65 · 0.05–0.90 · smooth" or "γ 1.00 · 0.00–1.00 · 8 steps · inverted"."""
    parts = [f"γ {mapping['gamma']:.2f}",
             f"{mapping['dark_level']:.2f}–{mapping['light_level']:.2f}",
             f"{mapping['steps']} steps" if mapping["steps"] >= 2 else "smooth"]
    if mapping["invert"]:
        parts.append("inverted")
    return " · ".join(parts)


def changed_parameters(reference: dict, other: dict) -> set[str]:
    """Return the config parameter names whose values differ; dark/light_level count as gray_levels."""
    return {MAPPING_PARAMETERS.get(key, key)
            for key in set(reference) | set(other)
            if reference.get(key) != other.get(key)}
