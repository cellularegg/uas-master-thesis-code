"""MLflow helpers for sealed-test water-level regimes.

Regime diagnostics are deliberately kept out of candidate selection.  This
module only normalises the descriptive tables produced after a model has been
selected and scored on its sealed-test cohort, and provides one stable naming
scheme for MLflow.
"""

from collections.abc import Mapping
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pandas as pd

from src.config import (
    WATER_LEVEL_ALARM_CONFIG_VERSION,
    WATER_LEVEL_ALARM_THRESHOLD_CM,
    WATER_LEVEL_ALARM_THRESHOLD_SOURCE,
    WATER_LEVEL_ALARM_THRESHOLD_STATION_ID,
)

REGIME_METRIC_NAMES = ("mae", "rmse", "me")
REGIME_DEFINITION_VERSION = "1"
DEFAULT_ALARM_THRESHOLD_CM = WATER_LEVEL_ALARM_THRESHOLD_CM
DEFAULT_ALARM_THRESHOLD_SOURCE = WATER_LEVEL_ALARM_THRESHOLD_SOURCE


def build_regime_definition(
    quartile_cutoffs_cm: tuple[float, float, float],
    quartile_reference_count: int,
    *,
    alarm_threshold_cm: float = DEFAULT_ALARM_THRESHOLD_CM,
) -> dict[str, object]:
    """Build the versioned regime definition stored with an execution.

    Args:
        quartile_cutoffs_cm: Training-derived Q25, Q50, and Q75 cutoffs.
        quartile_reference_count: Number of finite, non-imputed training
            observations used to derive the cutoffs.
        alarm_threshold_cm: Snapshot of the PegelAlarm alarm value.

    Returns:
        JSON-safe regime definition.

    Raises:
        ValueError: If the cutoffs, reference count, or threshold are invalid.
    """
    if len(quartile_cutoffs_cm) != 3:
        raise ValueError("Exactly three quartile cutoffs are required")
    cutoffs = tuple(float(value) for value in quartile_cutoffs_cm)
    if not all(np.isfinite(value) for value in cutoffs):
        raise ValueError("Quartile cutoffs must be finite")
    if any(lower > upper for lower, upper in pairwise(cutoffs)):
        raise ValueError("Quartile cutoffs must be nondecreasing")
    if int(quartile_reference_count) < 1:
        raise ValueError("Quartile reference count must be positive")
    threshold = float(alarm_threshold_cm)
    if not np.isfinite(threshold):
        raise ValueError("Alarm threshold must be finite")
    return {
        "version": REGIME_DEFINITION_VERSION,
        "quartile_cutoffs_cm": list(cutoffs),
        "quartile_reference_count": int(quartile_reference_count),
        "alarm_threshold_cm": threshold,
        "alarm_threshold_source": DEFAULT_ALARM_THRESHOLD_SOURCE,
        "alarm_config_version": int(WATER_LEVEL_ALARM_CONFIG_VERSION),
        "alarm_threshold_station_id": WATER_LEVEL_ALARM_THRESHOLD_STATION_ID,
    }


def normalize_regime_definition(
    definition: Mapping[str, Any],
) -> dict[str, object]:
    """Validate and normalise a stored regime definition.

    Both the public tuple spelling (``quartile_cutoffs_cm``) and the compact
    mapping spelling (``q25_cm``/``q50_cm``/``q75_cm``) are accepted.  The
    latter keeps this boundary tolerant of older notebook call sites while
    MLflow uses one canonical representation.

    Args:
        definition: Mapping to validate.

    Returns:
        Canonical JSON-safe regime definition.

    Raises:
        ValueError: If required provenance is missing or inconsistent.
    """
    raw_cutoffs = definition.get("quartile_cutoffs_cm")
    if raw_cutoffs is None:
        raw_cutoffs = (
            definition.get("q25_cm"),
            definition.get("q50_cm"),
            definition.get("q75_cm"),
        )
    if not isinstance(raw_cutoffs, (list, tuple)):
        raise TypeError("Regime definition quartile_cutoffs_cm must be a sequence")
    if any(value is None for value in raw_cutoffs):
        raise ValueError("Regime definition is missing quartile cutoffs")
    reference_count = definition.get("quartile_reference_count")
    if reference_count is None:
        reference_count = definition.get("reference_count")
    if reference_count is None:
        raise ValueError("Regime definition is missing quartile reference count")
    threshold = definition.get("alarm_threshold_cm", DEFAULT_ALARM_THRESHOLD_CM)
    cutoff_values = tuple(float(cast(Any, value)) for value in raw_cutoffs)
    if len(cutoff_values) != 3:
        raise ValueError("Exactly three quartile cutoffs are required")
    cutoffs = (cutoff_values[0], cutoff_values[1], cutoff_values[2])
    normalised = build_regime_definition(
        cutoffs,
        int(reference_count),
        alarm_threshold_cm=float(threshold),
    )
    if "version" in definition:
        normalised["version"] = str(definition["version"])
    if "alarm_threshold_source" in definition:
        normalised["alarm_threshold_source"] = str(definition["alarm_threshold_source"])
    if "alarm_config_version" in definition:
        normalised["alarm_config_version"] = int(definition["alarm_config_version"])
    if "alarm_threshold_station_id" in definition:
        normalised["alarm_threshold_station_id"] = str(
            definition["alarm_threshold_station_id"]
        )
    if "station_id" in definition:
        normalised["station_id"] = str(definition["station_id"])
    return normalised


def _regime_slug(regime: object) -> str:
    """Return the stable MLflow slug for one regime label."""
    text = str(regime).strip().lower()
    return {"q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4", "alarm": "alarm"}.get(
        text, text.replace(" ", "_")
    )


def _row_count(row: Mapping[Any, Any]) -> int:
    """Read the count field from either supported regime-table spelling."""
    for key in ("scored_values", "scored_forecast_values", "sample_count", "count"):
        value = row.get(key)
        if value is not None and pd.notna(value):
            return int(value)
    return 0


def regime_mlflow_metrics(
    aggregate_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
) -> dict[str, float]:
    """Flatten regime aggregate and per-horizon rows into stable MLflow names.

    Aggregate metrics are named ``regime_<slug>_<metric>`` and
    ``regime_<slug>_scored_values``.  Horizon metrics add
    ``_horizon_NN`` before the metric/count suffix.  Empty regime cells retain
    their zero count and simply omit unavailable error metrics.

    Args:
        aggregate_metrics: One row per regime, without a horizon column.
        horizon_metrics: One row per regime and horizon.

    Returns:
        MLflow-compatible scalar metric mapping.

    Raises:
        ValueError: If a required regime or horizon column is missing.
    """
    result: dict[str, float] = {}
    for frame, per_horizon in ((aggregate_metrics, False), (horizon_metrics, True)):
        required = {"regime"}
        if per_horizon:
            required.add("horizon_hours")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Regime metrics are missing columns: {missing}")
        for row in frame.to_dict(orient="records"):
            slug = _regime_slug(row["regime"])
            prefix = f"regime_{slug}"
            if per_horizon:
                prefix += f"_horizon_{int(row['horizon_hours']):02d}"
            result[f"{prefix}_scored_values"] = float(_row_count(row))
            for metric in REGIME_METRIC_NAMES:
                value = row.get(metric)
                if value is not None and pd.notna(value) and np.isfinite(float(value)):
                    result[f"{prefix}_{metric}"] = float(value)
    return result


def regime_mlflow_params(definition: Mapping[str, Any]) -> dict[str, object]:
    """Return stable MLflow parameters for regime cutoff provenance."""
    canonical = normalize_regime_definition(definition)
    cutoffs = canonical["quartile_cutoffs_cm"]
    assert isinstance(cutoffs, list)
    return {
        "regime_definition_version": str(canonical["version"]),
        "regime_q25_cm": float(cutoffs[0]),
        "regime_q50_cm": float(cutoffs[1]),
        "regime_q75_cm": float(cutoffs[2]),
        "regime_quartile_reference_count": int(
            cast(Any, canonical["quartile_reference_count"])
        ),
        "regime_alarm_threshold_cm": float(cast(Any, canonical["alarm_threshold_cm"])),
        "regime_alarm_threshold_station_id": str(
            canonical["alarm_threshold_station_id"]
        ),
        "regime_alarm_config_version": int(
            cast(Any, canonical["alarm_config_version"])
        ),
        "regime_alarm_threshold_source": str(canonical["alarm_threshold_source"]),
    }


def sealed_test_regime_tables(
    actual: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_columns: tuple[str, ...] | list[str],
    station_id: str,
    quartile_cutoffs_cm: tuple[float, float, float],
    quartile_reference_count: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Calculate sealed-test regime tables and their provenance definition.

    The import is local to avoid a module cycle while the core metric
    implementation is being loaded.

    Args:
        actual: Sealed-test actual target frame.
        predictions: Sealed-test predictions.
        target_columns: Ordered direct-forecast target columns.
        station_id: Target station identifier.
        quartile_cutoffs_cm: Training-derived quartile cutoffs.
        quartile_reference_count: Training reference population size.

    Returns:
        Definition, aggregate regime table, and per-horizon regime table.

    Raises:
        ValueError: If the configured alarm threshold is for another station or
            the regime tables reject the supplied inputs.
    """
    if station_id != WATER_LEVEL_ALARM_THRESHOLD_STATION_ID:
        raise ValueError(
            "The configured water-level alarm snapshot belongs to "
            f"{WATER_LEVEL_ALARM_THRESHOLD_STATION_ID!r}, not {station_id!r}"
        )
    from src.metrics import water_level_regime_tables

    definition = build_regime_definition(
        quartile_cutoffs_cm,
        quartile_reference_count,
        alarm_threshold_cm=WATER_LEVEL_ALARM_THRESHOLD_CM,
    )
    aggregate, horizon = water_level_regime_tables(
        actual,
        predictions,
        target_columns=target_columns,
        station_id=station_id,
        quartile_cutoffs_cm=quartile_cutoffs_cm,
        alarm_threshold_cm=WATER_LEVEL_ALARM_THRESHOLD_CM,
    )
    return definition, aggregate, horizon
