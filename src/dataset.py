"""The joined feature dataset every stage-4 model is fit and scored on."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]


@dataclass(frozen=True)
class JoinedFeatureContract:
    """Ordered feature contract for one engineered target station.

    Attributes:
        station_id: Identifier of the station whose targets are predicted.
        target_valid_column: Column marking rows with a complete target horizon.
        predictor_columns: Ordered predictor columns declared by feature metadata.
        target_columns: Ordered direct-forecast target columns.
    """

    station_id: str
    target_valid_column: str
    predictor_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class JoinedDataset:
    """Everything a stage-4 notebook needs from the joined feature artifacts.

    Attributes:
        contract: Validated predictor, target, and eligibility contract.
        train_rows: Eligible training cohort in chronological issue-time order.
        test_rows: Eligible sealed-test cohort in chronological order.
        feature_subsets: The named ablation subsets, in contract order.
        folds: Materialized expanding validation folds over ``train_rows``.
        validation_test_size: Rows in each fold's validation window.
        input_hashes: SHA-256 digests of the train and test Parquet inputs.
        raw_row_counts: Row counts of the unfiltered train and test artifacts.
        target_context_series: Timestamp-indexed target-station water level and
            imputation flag across both artifacts.
        target_water_level_quartile_cutoffs_cm: Training-reference Q25, Q50,
            and Q75 target water-level cutoffs in centimetres.
        target_water_level_quartile_reference_count: Number of finite,
            non-imputed training measurements used for the quartiles.
    """

    contract: JoinedFeatureContract
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    feature_subsets: dict[str, list[str]]
    folds: list[tuple[np.ndarray, np.ndarray]]
    validation_test_size: int
    input_hashes: dict[str, str]
    raw_row_counts: dict[str, int]
    target_context_series: pd.DataFrame
    target_water_level_quartile_cutoffs_cm: tuple[float, float, float]
    target_water_level_quartile_reference_count: int


def load_joined_dataset(
    metadata_path: Path,
    train_path: Path,
    test_path: Path,
    *,
    station_id: str,
    forecast_horizon_hours: int,
    weather_variables: Sequence[str],
    initial_train_fraction: float,
    n_validation_folds: int,
    embargo_rows: int,
) -> JoinedDataset:
    """Load, validate, and prepare the joined feature dataset in one call.

    Reads the joined metadata contract and both Parquet artifacts, filters each
    to the common eligible cohort, derives the ablation subsets and the
    chronological validation folds, and collects the provenance a stage-4
    notebook logs alongside its run.

    Args:
        metadata_path: Joined-feature metadata JSON path.
        train_path: Joined training-feature Parquet path.
        test_path: Joined sealed-test-feature Parquet path.
        station_id: Target station expected in the metadata.
        forecast_horizon_hours: Required direct-forecast horizon.
        weather_variables: Base names treated as raw weather predictors.
        initial_train_fraction: Fraction reserved before validation allocation.
        n_validation_folds: Number of expanding validation folds.
        embargo_rows: Rows excluded between each training and validation window.

    Returns:
        The prepared dataset.

    Raises:
        FileNotFoundError: If the metadata, train, or test artifact is missing.
        ValueError: If any artifact violates the expected contract, a cohort is
            empty, a subset is empty, or the folds violate the CV policy.
    """
    contract, train_features, test_features = _load_joined_training_data(
        metadata_path,
        train_path,
        test_path,
        station_id=station_id,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    train_rows = _prepare_model_rows(train_features, contract, artifact_name="train")
    test_rows = _prepare_model_rows(test_features, contract, artifact_name="test")
    folds, validation_test_size = time_series_splits(
        len(train_rows),
        initial_train_fraction=initial_train_fraction,
        n_validation_folds=n_validation_folds,
        embargo_rows=embargo_rows,
    )
    feature_subsets = _build_feature_subsets(
        contract, weather_variables=weather_variables
    )
    target_context_series = _target_context_series(
        train_features, test_features, contract
    )
    quartile_cutoffs, quartile_reference_count = _target_water_level_quartile_summary(
        train_features, contract
    )
    return JoinedDataset(
        contract=contract,
        train_rows=train_rows,
        test_rows=test_rows,
        feature_subsets=feature_subsets,
        folds=folds,
        validation_test_size=validation_test_size,
        input_hashes={
            "train_input_sha256": _sha256_file(train_path),
            "test_input_sha256": _sha256_file(test_path),
        },
        raw_row_counts={"train": len(train_features), "test": len(test_features)},
        target_context_series=target_context_series,
        target_water_level_quartile_cutoffs_cm=quartile_cutoffs,
        target_water_level_quartile_reference_count=quartile_reference_count,
    )


def _load_joined_training_data(
    metadata_path: Path,
    train_path: Path,
    test_path: Path,
    *,
    station_id: str,
    forecast_horizon_hours: int,
) -> tuple[JoinedFeatureContract, pd.DataFrame, pd.DataFrame]:
    """Load a station contract and its joined train and test artifacts.

    Args:
        metadata_path: Joined-feature metadata JSON path.
        train_path: Joined training-feature Parquet path.
        test_path: Joined sealed-test-feature Parquet path.
        station_id: Target station expected in the metadata.
        forecast_horizon_hours: Required direct-forecast horizon.

    Returns:
        The validated station contract, training frame, and sealed-test frame.

    Raises:
        FileNotFoundError: If the metadata, train, or test artifact is missing.
        ValueError: If metadata or either frame violates the expected contract.
    """
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing joined feature artifact: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("configuration", {}).get("horizon_hours") != forecast_horizon_hours:
        raise ValueError(
            "Feature metadata horizon does not match forecast_horizon_hours"
        )
    if station_id not in metadata.get("engineered_station_ids", []):
        raise ValueError(
            f"Target station {station_id!r} is not engineered in feature metadata"
        )
    expected_targets = tuple(
        f"{station_id}__target_t_plus_{offset:02d}"
        for offset in range(1, forecast_horizon_hours + 1)
    )
    if tuple(metadata.get("target_columns", ())) != expected_targets:
        raise ValueError(
            "Feature metadata target columns do not match the configured horizon"
        )
    predictor_columns = tuple(metadata.get("predictor_columns", ()))
    if not predictor_columns:
        raise ValueError("The metadata predictor contract is empty")
    contract = JoinedFeatureContract(
        station_id=station_id,
        target_valid_column=f"{station_id}__target_valid",
        predictor_columns=predictor_columns,
        target_columns=expected_targets,
    )
    for artifact_path in (train_path, test_path):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing joined feature artifact: {artifact_path}")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    _validate_frame_contract(train, contract, artifact_name="train")
    _validate_frame_contract(test, contract, artifact_name="test")
    return contract, train, test


def _validate_frame_contract(
    frame: pd.DataFrame,
    contract: JoinedFeatureContract,
    *,
    artifact_name: str,
) -> None:
    required_columns = {
        "timestamp",
        contract.target_valid_column,
        *contract.predictor_columns,
        *contract.target_columns,
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{artifact_name} artifact is missing required columns: {missing}"
        )


def _prepare_model_rows(
    frame: pd.DataFrame,
    contract: JoinedFeatureContract,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """Return the complete common cohort in chronological issue-time order.

    Args:
        frame: Joined feature frame to filter.
        contract: Predictor, target, and eligibility-column contract.
        artifact_name: Human-readable artifact label used in validation errors.

    Returns:
        Eligible rows sorted by timestamp with a fresh integer index.

    Raises:
        ValueError: If required columns are missing, target-valid rows have null
            targets, or no eligible rows remain.
    """
    _validate_frame_contract(frame, contract, artifact_name=artifact_name)
    target_valid_rows = frame[contract.target_valid_column].eq(True)
    if (
        frame.loc[target_valid_rows, list(contract.target_columns)]
        .isna()
        .any(axis=None)
    ):
        raise ValueError(
            f"{artifact_name} artifact has null targets in target-valid rows"
        )
    eligible = (
        target_valid_rows
        & frame[list(contract.predictor_columns)].notna().all(axis=1)
        & frame[list(contract.target_columns)].notna().all(axis=1)
    )
    rows = (
        frame.loc[eligible]
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    if rows.empty:
        raise ValueError(f"{artifact_name} artifact has no eligible model rows")
    return rows


def _feature_parts(column: str) -> tuple[str, str]:
    station_id, separator, base_name = column.partition("__")
    if not separator or not station_id or not base_name:
        raise ValueError(f"Predictor {column!r} must use '<station>__<feature>' format")
    return station_id, base_name


def _is_hydrology_quality_or_time(base_name: str) -> bool:
    return base_name in {"water_level", "imputed"} or base_name.startswith(
        ("water_level_", "imputed_count_", "utc_")
    )


def _build_feature_subsets(
    contract: JoinedFeatureContract,
    *,
    weather_variables: Sequence[str],
) -> dict[str, list[str]]:
    """Build six ordered ablation subsets from a joined predictor contract.

    Args:
        contract: Ordered joined-feature contract for the target station.
        weather_variables: Base names treated as raw weather predictors.

    Returns:
        The six named feature subsets in contract order.

    Raises:
        ValueError: If predictors are duplicated or malformed, or a required
            subset is empty.
    """
    if len(contract.predictor_columns) != len(set(contract.predictor_columns)):
        raise ValueError("The predictor contract contains duplicates")
    parsed_columns = [
        (column, *_feature_parts(column)) for column in contract.predictor_columns
    ]
    raw_names = {"water_level", "imputed", *weather_variables}
    subsets = {
        "full": list(contract.predictor_columns),
        "all_station_hydrology_quality_time": [
            column
            for column, _station_id, base_name in parsed_columns
            if _is_hydrology_quality_or_time(base_name)
        ],
        "raw_all_stations": [
            column
            for column, _station_id, base_name in parsed_columns
            if base_name in raw_names
        ],
        "target_station_full": [
            column
            for column, station_id, _base_name in parsed_columns
            if station_id == contract.station_id
        ],
        "target_station_hydrology_quality_time": [
            column
            for column, station_id, base_name in parsed_columns
            if station_id == contract.station_id
            and _is_hydrology_quality_or_time(base_name)
        ],
        "current_water_levels_all_stations": [
            column
            for column, _station_id, base_name in parsed_columns
            if base_name == "water_level"
        ],
    }
    for subset_name, columns in subsets.items():
        if not columns:
            raise ValueError(f"Feature subset {subset_name!r} is empty")
    return subsets


def time_series_splits(
    n_rows: int,
    *,
    initial_train_fraction: float,
    n_validation_folds: int,
    embargo_rows: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Build expanding chronological validation folds with an embargo.

    Args:
        n_rows: Number of eligible chronological training rows.
        initial_train_fraction: Fraction reserved before validation allocation.
        n_validation_folds: Number of expanding validation folds.
        embargo_rows: Rows excluded between each training and validation window.

    Returns:
        The materialized folds and the validation-window size.

    Raises:
        ValueError: If there are too few rows or the generated folds violate the
            configured count, chronology, embargo, or non-overlap invariants.
    """
    initial_train_rows = int(n_rows * initial_train_fraction)
    validation_budget = n_rows - initial_train_rows - embargo_rows
    validation_test_size = validation_budget // n_validation_folds
    if validation_test_size < 1:
        raise ValueError(
            "Not enough eligible training rows for the configured CV policy"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_validation_folds,
        gap=embargo_rows,
        test_size=validation_test_size,
    )
    splits = list(splitter.split(np.arange(n_rows)))
    if len(splits) != n_validation_folds:
        raise ValueError(
            f"Expected {n_validation_folds} validation folds, got {len(splits)}"
        )

    previous_validation_end = -1
    for fold_number, (fold_train_indices, fold_validation_indices) in enumerate(
        splits, start=1
    ):
        if fold_train_indices.size == 0 or fold_validation_indices.size == 0:
            raise ValueError(f"Fold {fold_number} is empty")
        if not np.array_equal(fold_train_indices, np.arange(fold_train_indices.size)):
            raise ValueError(f"Fold {fold_number} training rows are not chronological")
        if fold_validation_indices[0] - fold_train_indices[-1] - 1 != embargo_rows:
            raise ValueError(f"Fold {fold_number} does not have the configured embargo")
        if fold_validation_indices[0] <= previous_validation_end:
            raise ValueError("Validation folds overlap or are out of order")
        previous_validation_end = fold_validation_indices[-1]
    return splits, validation_test_size


def _target_context_series(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    contract: JoinedFeatureContract,
) -> pd.DataFrame:
    """Build the timestamp-indexed target-station ground-truth context.

    Args:
        train_features: Unfiltered joined training frame.
        test_features: Unfiltered joined sealed-test frame.
        contract: Joined-feature contract for the target station.

    Returns:
        The target station's water level and imputation flag across both
        artifacts, indexed by unique UTC timestamp in chronological order.

    Raises:
        ValueError: If the contract does not declare the target station's raw
            water-level and imputation columns.
    """
    water_level_column = f"{contract.station_id}__water_level"
    imputed_column = f"{contract.station_id}__imputed"
    missing = sorted(
        {water_level_column, imputed_column}.difference(contract.predictor_columns)
    )
    if missing:
        raise ValueError(
            f"The predictor contract is missing target-station context columns: "
            f"{missing}"
        )
    context_columns = ["timestamp", water_level_column, imputed_column]
    context = pd.concat(
        [train_features[context_columns], test_features[context_columns]],
        ignore_index=True,
    )
    context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
    return (
        context.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
        .sort_index()
    )


def _target_water_level_quartile_summary(
    train_features: pd.DataFrame,
    contract: JoinedFeatureContract,
) -> tuple[tuple[float, float, float], int]:
    """Calculate training-reference target water-level quartile cutoffs.

    The reference population is deliberately independent of the model cohort:
    it uses every finite, non-imputed target-station water-level measurement in
    the unfiltered training artifact, including rows that are later excluded
    from modelling because another feature or target is incomplete.

    Args:
        train_features: Unfiltered joined training feature artifact.
        contract: Validated joined-feature contract.

    Returns:
        The Q25, Q50, and Q75 cutoffs in centimetres and the number of reference
        measurements used to calculate them.

    Raises:
        ValueError: If the target-station context columns are missing, no valid
            reference measurements exist, or the resulting cutoffs are not
            finite and ordered.
    """
    water_level_column = f"{contract.station_id}__water_level"
    imputed_column = f"{contract.station_id}__imputed"
    missing = sorted(
        {water_level_column, imputed_column}.difference(train_features.columns)
    )
    if missing:
        raise ValueError(
            f"The training artifact is missing target-station context columns: "
            f"{missing}"
        )

    water_levels = pd.to_numeric(
        train_features[water_level_column], errors="coerce"
    ).to_numpy(dtype=float)
    non_imputed = (
        train_features[imputed_column].eq(False).fillna(False).to_numpy(dtype=bool)
    )
    reference_values = water_levels[np.isfinite(water_levels) & non_imputed]
    reference_count = int(reference_values.size)
    if reference_count == 0:
        raise ValueError(
            "The training artifact has no finite, non-imputed target water-level "
            "measurements"
        )

    cutoffs_array = np.quantile(reference_values, [0.25, 0.50, 0.75])
    cutoffs: tuple[float, float, float] = (
        float(cutoffs_array[0]),
        float(cutoffs_array[1]),
        float(cutoffs_array[2]),
    )
    if not all(np.isfinite(cutoff) for cutoff in cutoffs) or any(
        lower > upper for lower, upper in pairwise(cutoffs)
    ):
        raise ValueError(
            "Target water-level quartile cutoffs must be finite and ordered"
        )
    return cutoffs, reference_count


def _sha256_file(path: Path) -> str:
    """Compute the chunked SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        The hex-encoded SHA-256 digest, used for MLflow provenance params.
    """
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
