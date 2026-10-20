import numpy as np
import pandas as pd
import pytest

from anomaly_engine import (
    DataValidationError,
    REQUIRED_COLUMNS,
    score_anomalies,
    validate_readings,
)


def make_data(components=12, points=5):
    rows = []
    for c in range(components):
        for t in range(points):
            rows.append([
                f"C{c:03d}",
                t * 2,
                40 + 0.2 * t + c * 0.01,
                0.8 + 0.001 * t,
                5.0 + (0.001 if t % 2 else -0.001),
            ])
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_valid_data_is_sorted_and_reported():
    data = make_data()
    data = data.sample(frac=1, random_state=5)
    cleaned, report = validate_readings(data)
    assert report.rows_received == len(data)
    assert report.rows_used == len(data)
    assert report.components == 12
    assert cleaned.groupby("component_id")["time_min"].apply(lambda s: s.is_monotonic_increasing).all()


def test_missing_column_fails_clearly():
    data = make_data().drop(columns=["voltage_v"])
    with pytest.raises(DataValidationError, match="voltage_v"):
        validate_readings(data)


def test_non_numeric_values_are_not_silently_dropped():
    data = make_data()
    data.loc[0, "current_a"] = "not-a-number"
    with pytest.raises(DataValidationError, match="current_a"):
        validate_readings(data)


def test_duplicate_component_time_fails():
    data = make_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_readings(data)


def test_single_reading_component_fails():
    data = make_data()
    data = data[data["component_id"] != "C000"]
    data = pd.concat([data, pd.DataFrame([["C000", 0, 40, .8, 5.0]], columns=REQUIRED_COLUMNS)])
    with pytest.raises(DataValidationError, match="at least 2"):
        validate_readings(data)


def test_scoring_is_deterministic_and_has_expected_columns():
    clean, _ = validate_readings(make_data())
    _, first, threshold1 = score_anomalies(clean)
    _, second, threshold2 = score_anomalies(clean)
    assert first["component_id"].tolist() == second["component_id"].tolist()
    assert np.allclose(first["anomaly_score"], second["anomaly_score"])
    assert threshold1 == pytest.approx(threshold2)
    assert {"status", "anomaly_score", "temperature_change_c"} <= set(first.columns)


def test_constant_features_do_not_produce_nan_or_infinite_scores():
    rows = []
    for c in range(12):
        for t in range(3):
            rows.append([f"C{c:03d}", t, 40.0, 0.8, 5.0])
    clean, _ = validate_readings(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))
    _, summary, threshold = score_anomalies(clean)
    assert np.isfinite(summary["anomaly_score"]).all()
    assert np.isfinite(threshold)
    assert (summary["status"] == "Potential anomaly").sum() == 0


def test_invalid_percentile_fails():
    clean, _ = validate_readings(make_data())
    with pytest.raises(ValueError, match="percentile"):
        score_anomalies(clean, percentile=100)
