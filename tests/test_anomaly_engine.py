import numpy as np
import pandas as pd
import pytest

from anomaly_engine import (
    DataValidationError,
    FEATURE_COLUMNS,
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
    data = make_data().sample(frac=1, random_state=5)
    cleaned, report = validate_readings(data)
    assert report.rows_received == len(data)
    assert report.rows_used == len(data)
    assert report.components == 12
    assert cleaned.groupby("component_id")["time_min"].apply(lambda s: s.is_monotonic_increasing).all()


def test_missing_column_fails_clearly():
    with pytest.raises(DataValidationError, match="voltage_v"):
        validate_readings(make_data().drop(columns=["voltage_v"]))


def test_non_numeric_values_are_not_silently_dropped():
    data = make_data()
    data.loc[0, "current_a"] = "not-a-number"
    with pytest.raises(DataValidationError, match="current_a"):
        validate_readings(data)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(bad_value):
    data = make_data()
    data.loc[0, "temperature_c"] = bad_value
    with pytest.raises(DataValidationError, match="temperature_c"):
        validate_readings(data)


def test_duplicate_component_time_fails():
    data = pd.concat([make_data(), make_data().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_readings(data)


def test_duplicate_column_names_fail():
    data = make_data()
    data.columns = list(data.columns[:-1]) + ["current_a"]
    with pytest.raises(DataValidationError, match="Duplicate column"):
        validate_readings(data)


def test_single_reading_component_fails():
    data = make_data()
    data = data[data["component_id"] != "C000"]
    data = pd.concat(
        [data, pd.DataFrame([["C000", 0, 40, .8, 5.0]], columns=REQUIRED_COLUMNS)],
        ignore_index=True,
    )
    with pytest.raises(DataValidationError, match="at least 2"):
        validate_readings(data)


def test_empty_data_fails():
    with pytest.raises(DataValidationError, match="empty"):
        validate_readings(pd.DataFrame(columns=REQUIRED_COLUMNS))


def test_negative_time_fails():
    data = make_data()
    data.loc[0, "time_min"] = -1
    with pytest.raises(DataValidationError, match="cannot be negative"):
        validate_readings(data)


def test_component_id_whitespace_is_trimmed():
    data = make_data()
    data.loc[data["component_id"] == "C000", "component_id"] = " C000 "
    clean, _ = validate_readings(data)
    assert "C000" in set(clean["component_id"])
    assert " C000 " not in set(clean["component_id"])


def test_scoring_is_deterministic_and_has_expected_columns():
    clean, _ = validate_readings(make_data())
    _, first, threshold1 = score_anomalies(clean)
    _, second, threshold2 = score_anomalies(clean)
    assert first["component_id"].tolist() == second["component_id"].tolist()
    assert np.allclose(first["anomaly_score"], second["anomaly_score"])
    assert threshold1 == pytest.approx(threshold2)
    assert {"status", "anomaly_score", "temperature_change_c"} <= set(first.columns)
    assert all(f"deviation_{feature}" in first.columns for feature in FEATURE_COLUMNS)


def test_constant_features_do_not_produce_nan_or_infinite_scores():
    rows = [
        [f"C{c:03d}", t, 40.0, 0.8, 5.0]
        for c in range(12) for t in range(3)
    ]
    clean, _ = validate_readings(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))
    _, summary, threshold = score_anomalies(clean)
    assert np.isfinite(summary["anomaly_score"]).all()
    assert np.isfinite(threshold)
    assert (summary["status"].str.startswith("Potential anomaly")).sum() == 0


def test_injected_multisignal_outlier_ranks_first():
    data = make_data(components=20, points=10)
    mask = data["component_id"] == "C019"
    times = data.loc[mask, "time_min"].to_numpy()
    data.loc[mask, "temperature_c"] += np.maximum(times - 10, 0) * 1.5
    data.loc[mask, "current_a"] += np.maximum(times - 10, 0) * 0.08
    clean, _ = validate_readings(data)
    _, summary, _ = score_anomalies(clean, percentile=90)
    assert summary.iloc[0]["component_id"] == "C019"
    assert summary.iloc[0]["anomaly_score"] > 0


def test_sampling_gap_warning_is_reported():
    data = make_data()
    data.loc[(data["component_id"] == "C000") & (data["time_min"] == 4), "time_min"] = 100
    _, report = validate_readings(data)
    assert any("sampling gaps" in warning for warning in report.warnings)


@pytest.mark.parametrize("percentile", [100, 0, np.nan, np.inf])
def test_invalid_percentile_fails(percentile):
    clean, _ = validate_readings(make_data())
    with pytest.raises(ValueError, match="percentile"):
        score_anomalies(clean, percentile=percentile)
