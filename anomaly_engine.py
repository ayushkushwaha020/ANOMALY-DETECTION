"""Validation and explainable anomaly screening for burn-in readings.

This is a transparent statistical heuristic, not a trained or validated
failure-prediction model. It must not be used as the sole pass/fail gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "component_id",
    "time_min",
    "temperature_c",
    "current_a",
    "voltage_v",
)
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
FEATURE_COLUMNS = (
    "temperature_change_c",
    "current_change_a",
    "voltage_std_v",
    "temperature_rate_max_c_per_min",
    "current_rate_max_a_per_min",
    "voltage_rate_max_v_per_min",
    "temperature_std_c",
    "current_std_a",
    "voltage_change_v",
    "temperature_peak_to_peak_c",
    "current_peak_to_peak_a",
    "voltage_peak_to_peak_v",
    "sampling_gap_max_min",
)


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be analyzed safely."""


@dataclass(frozen=True)
class ValidationReport:
    rows_received: int
    rows_used: int
    components: int
    warnings: tuple[str, ...] = ()


def validate_readings(data: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate required schema and values; never silently discard bad readings."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Input must be a CSV-like table.")
    if data.empty:
        raise DataValidationError("The dataset is empty.")
    if len(data) > 1_000_000:
        raise DataValidationError("Dataset exceeds the 1,000,000-row safety limit.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].astype(str).tolist()
        raise DataValidationError("Duplicate column names found: " + ", ".join(duplicates))

    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        raise DataValidationError("Missing required column(s): " + ", ".join(missing))

    frame = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["component_id"] = frame["component_id"].astype("string").str.strip()
    if frame["component_id"].isna().any() or frame["component_id"].eq("").any():
        raise DataValidationError("Every row must have a non-empty component_id.")
    if frame["component_id"].str.len().gt(128).any():
        raise DataValidationError("component_id values must be 128 characters or fewer.")

    for name in NUMERIC_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        values = frame[name].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            bad_count = int((~np.isfinite(values)).sum())
            raise DataValidationError(
                f"Column '{name}' has {bad_count} missing, non-numeric, or infinite value(s). "
                "Correct the CSV and upload it again."
            )

    if (frame["time_min"] < 0).any():
        raise DataValidationError("time_min cannot be negative.")

    duplicate_mask = frame.duplicated(["component_id", "time_min"], keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["component_id", "time_min"]].head(3)
        pairs = ", ".join(f"{r.component_id}@{r.time_min:g}" for r in examples.itertuples())
        raise DataValidationError(
            f"Duplicate component/time readings found (for example: {pairs}). "
            "Each component must have at most one reading per time_min."
        )

    counts = frame.groupby("component_id", sort=False).size()
    too_short = counts[counts < 2]
    if not too_short.empty:
        ids = ", ".join(map(str, too_short.index[:5]))
        raise DataValidationError(
            f"Each component needs at least 2 time readings. Insufficient data for: {ids}."
        )

    frame = frame.sort_values(["component_id", "time_min"], kind="stable").reset_index(drop=True)
    warnings: list[str] = []
    component_count = int(frame["component_id"].nunique())
    if component_count < 10:
        warnings.append(
            "Fewer than 10 components are available. Relative anomaly rankings may be unstable."
        )

    gaps = frame.groupby("component_id", sort=False)["time_min"].diff().dropna()
    if not gaps.empty:
        median_gap = float(gaps.median())
        if median_gap > 0 and bool((gaps > median_gap * 5).any()):
            warnings.append(
                "Large sampling gaps detected (more than 5× the dataset median interval). "
                "Review test interruptions and sampling consistency."
            )
    for column in ("temperature_c", "current_a", "voltage_v"):
        if frame[column].nunique() == 1:
            warnings.append(
                f"'{column}' is constant across all readings. Confirm sensor/channel validity."
            )

    report = ValidationReport(
        rows_received=len(data),
        rows_used=len(frame),
        components=component_count,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return frame, report


def _robust_deviation(values: Iterable[float]) -> np.ndarray:
    """Absolute robust z-like deviation, with safe behavior for constant features."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return np.array([], dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad > 1e-12:
        scale = 1.4826 * mad
    else:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(array, dtype=float)
    result = np.abs((array - median) / scale)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def build_component_features(readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return readings with time rates and one feature row per component."""
    frame = readings.copy()
    grouped = frame.groupby("component_id", sort=False)
    time_delta = grouped["time_min"].diff()

    for source, target in (
        ("temperature_c", "temperature_rate_c_per_min"),
        ("current_a", "current_rate_a_per_min"),
        ("voltage_v", "voltage_rate_v_per_min"),
    ):
        delta = grouped[source].diff()
        rate = delta.div(time_delta.where(time_delta > 0))
        frame[target] = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features = frame.groupby("component_id", sort=False).agg(
        temperature_mean_c=("temperature_c", "mean"),
        temperature_max_c=("temperature_c", "max"),
        temperature_change_c=("temperature_c", lambda s: float(s.iloc[-1] - s.iloc[0])),
        temperature_std_c=("temperature_c", lambda s: float(s.std(ddof=0))),
        temperature_peak_to_peak_c=("temperature_c", lambda s: float(s.max() - s.min())),
        current_mean_a=("current_a", "mean"),
        current_change_a=("current_a", lambda s: float(s.iloc[-1] - s.iloc[0])),
        current_std_a=("current_a", lambda s: float(s.std(ddof=0))),
        current_peak_to_peak_a=("current_a", lambda s: float(s.max() - s.min())),
        voltage_std_v=("voltage_v", lambda s: float(s.std(ddof=0))),
        voltage_change_v=("voltage_v", lambda s: float(s.iloc[-1] - s.iloc[0])),
        voltage_peak_to_peak_v=("voltage_v", lambda s: float(s.max() - s.min())),
        temperature_rate_max_c_per_min=("temperature_rate_c_per_min", lambda s: float(s.abs().max())),
        current_rate_max_a_per_min=("current_rate_a_per_min", lambda s: float(s.abs().max())),
        voltage_rate_max_v_per_min=("voltage_rate_v_per_min", lambda s: float(s.abs().max())),
        sampling_gap_max_min=("time_min", lambda s: float(s.diff().max())),
        readings_count=("time_min", "size"),
        test_duration_min=("time_min", lambda s: float(s.max() - s.min())),
    ).reset_index()
    return frame, features


def score_anomalies(
    readings: pd.DataFrame, percentile: float = 90.0
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Score components by equal-weight robust deviation across selected features.

    A percentile cutoff controls triage volume; it is not a probability, error
    rate, or validated engineering threshold. Ties at the cutoff are not flagged.
    """
    if not np.isfinite(float(percentile)) or not 50.0 <= float(percentile) < 100.0:
        raise ValueError("percentile must be between 50 (inclusive) and 100 (exclusive).")
    if readings.empty:
        raise DataValidationError("No readings are available to score.")

    frame, features = build_component_features(readings)
    score_matrix = np.column_stack([
        _robust_deviation(features[column].to_numpy(dtype=float))
        for column in FEATURE_COLUMNS
    ])
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        features[f"deviation_{feature_name}"] = score_matrix[:, index]

    features["anomaly_score"] = score_matrix.mean(axis=1)
    threshold = float(np.percentile(features["anomaly_score"], float(percentile)))
    features["status"] = np.where(
        features["anomaly_score"] > threshold,
        "Potential anomaly — review required",
        "No statistical flag — pass status not determined",
    )
    features["screening_percentile"] = float(percentile)
    features = features.sort_values(
        ["anomaly_score", "component_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    return frame, features, threshold
