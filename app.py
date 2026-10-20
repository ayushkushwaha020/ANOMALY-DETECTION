from __future__ import annotations

from datetime import datetime, timezone
import io

import numpy as np
import pandas as pd
import streamlit as st

from anomaly_engine import (
    DataValidationError,
    REQUIRED_COLUMNS,
    score_anomalies,
    validate_readings,
)

st.set_page_config(
    page_title="CODE.KAISEN | Burn-in Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.0.0"

@st.cache_data(show_spinner=False)
def demo_data(components: int = 80, points: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create repeatable synthetic data for demos; this is not real test data."""
    rng = np.random.default_rng(seed)
    rows = []
    abnormal_indices = {11, 27, 53, 68}
    for c in range(components):
        component_id = f"C{c + 1:04d}"
        is_synthetic_outlier = c in abnormal_indices
        base_temp = rng.normal(48, 2)
        base_current = rng.normal(0.80, 0.05)
        base_voltage = rng.normal(5.0, 0.03)
        for t in range(points):
            temperature = base_temp + 0.18 * t + rng.normal(0, 0.35)
            current = base_current + 0.002 * t + rng.normal(0, 0.008)
            voltage = base_voltage + rng.normal(0, 0.008)
            if is_synthetic_outlier and t > 10:
                temperature += (t - 10) * 0.55
                current += (t - 10) * 0.010
                voltage += rng.normal(0, 0.035)
            rows.append([component_id, float(t), temperature, current, voltage])
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


with st.sidebar:
    st.markdown("## ⚡ CODE.KAISEN")
    st.caption("Burn-in & screening intelligence")
    st.divider()
    source = st.radio("Data source", ["Synthetic demo", "Upload CSV"], index=0)
    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload burn-in readings",
            type=["csv"],
            help="Required columns: " + ", ".join(REQUIRED_COLUMNS),
        )
        st.caption("CSV only. Max upload size is configured by the deployment.")
    percentile = st.slider(
        "Screening cutoff percentile",
        min_value=80,
        max_value=98,
        value=90,
        step=1,
        help="Higher cutoffs flag fewer components. This is a triage setting, not model accuracy.",
    )
    st.divider()
    st.markdown("**Analysis method**")
    st.caption("Robust statistical deviation across multi-parameter and time-rate features.")
    st.caption(f"Version {APP_VERSION}")

st.title("Component Burn-in Anomaly Detection")
st.caption("SIH PS 26170 • Explainable screening support • Engineer remains the decision-maker")

if source == "Upload CSV":
    if uploaded_file is None:
        st.info("Upload a CSV file to begin, or switch to Synthetic demo to explore the dashboard.")
        st.stop()
    try:
        raw_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    raw_data = demo_data()

try:
    readings, validation = validate_readings(raw_data)
    analyzed, summary, threshold = score_anomalies(readings, percentile=float(percentile))
except DataValidationError as exc:
    st.error(f"**Data validation failed:** {exc}")
    st.info("Check the CSV headers and values, then upload a corrected file. Invalid readings are not silently discarded.")
    st.stop()
except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
    st.error("The dataset could not be analyzed safely. Check the file format and numeric values, then retry.")
    st.caption(f"Diagnostic category: {type(exc).__name__}")
    st.stop()

flagged = summary.loc[summary["status"] == "Potential anomaly"].copy()
normal_count = int((summary["status"] == "No anomaly flagged").sum())

with st.container(border=True):
    st.markdown("#### Screening snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Components analyzed", f"{validation.components:,}")
    m2.metric("Flagged for review", f"{len(flagged):,}")
    m3.metric("Readings analyzed", f"{len(analyzed):,}")
    m4.metric("Cutoff score", fmt(threshold, 3))
    st.caption(
        f"Rows validated: {validation.rows_used:,} of {validation.rows_received:,} · "
        f"Cutoff: {percentile}th percentile · Generated: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

for warning in validation.warnings:
    st.warning(warning)

if source == "Synthetic demo":
    st.info(
        "You are viewing deterministic synthetic demonstration data. "
        "It does not represent real components or validated industrial test results."
    )

st.warning(
    "A potential anomaly is a review signal, not a confirmed defect or a prediction of failure. "
    "Do not use this prototype as the sole basis for pass/fail, safety, or production decisions."
)

overview_tab, flagged_tab, component_tab, trends_tab, export_tab = st.tabs(
    ["Overview", "Flagged components", "Component analysis", "Trends", "Export & data quality"]
)

with overview_tab:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Highest anomaly scores")
        view_cols = [
            "component_id", "status", "anomaly_score",
            "temperature_change_c", "current_change_a", "voltage_std_v",
        ]
        st.dataframe(
            summary[view_cols].head(15).rename(columns={
                "component_id": "Component ID",
                "status": "Screening status",
                "anomaly_score": "Anomaly score",
                "temperature_change_c": "Temperature change (°C)",
                "current_change_a": "Current change (A)",
                "voltage_std_v": "Voltage variation (V)",
            }),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Screening status")
        status_counts = summary["status"].value_counts().reindex(
            ["Potential anomaly", "No anomaly flagged"], fill_value=0
        )
        st.bar_chart(status_counts)
        st.caption("Counts depend on the uploaded dataset and selected cutoff.")

with flagged_tab:
    st.subheader("Potentially anomalous components")
    st.caption("Flagged for engineer review — these components are not confirmed defective.")
    display_columns = [
        "component_id", "anomaly_score", "temperature_change_c", "current_change_a",
        "voltage_std_v", "temperature_rate_max_c_per_min",
        "current_rate_max_a_per_min", "voltage_rate_max_v_per_min",
    ]
    flagged_view = flagged[display_columns].rename(columns={
        "component_id": "Component ID",
        "anomaly_score": "Anomaly score",
        "temperature_change_c": "Temperature change (°C)",
        "current_change_a": "Current change (A)",
        "voltage_std_v": "Voltage variation (V)",
        "temperature_rate_max_c_per_min": "Max |temperature rate| (°C/min)",
        "current_rate_max_a_per_min": "Max |current rate| (A/min)",
        "voltage_rate_max_v_per_min": "Max |voltage rate| (V/min)",
    })
    if flagged_view.empty:
        st.success("No components exceeded the selected cutoff. This does not prove all components are healthy.")
    else:
        st.dataframe(flagged_view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download flagged components CSV",
            data=to_csv_bytes(flagged_view),
            file_name="CODE.KAISEN_flagged_components.csv",
            mime="text/csv",
            use_container_width=False,
        )

with component_tab:
    component_ids = summary["component_id"].astype(str).tolist()
    selected_id = st.selectbox("Choose a component", component_ids)
    series = analyzed.loc[analyzed["component_id"].astype(str) == selected_id].sort_values("time_min")
    record = summary.loc[summary["component_id"].astype(str) == selected_id].iloc[0]
    x, y, z = st.columns(3)
    x.metric("Screening status", str(record["status"]))
    y.metric("Anomaly score", fmt(record["anomaly_score"], 3))
    z.metric("Cutoff score", fmt(threshold, 3))
    st.line_chart(series.set_index("time_min")[["temperature_c", "current_a", "voltage_v"]])
    st.caption("Parameters use different physical units; inspect individual trend charts before comparing magnitudes.")
    st.markdown("**Largest contributors to this component's score**")
    contribution_columns = [f"deviation_{name}" for name in (
        "temperature_change_c", "current_change_a", "voltage_std_v",
        "temperature_rate_max_c_per_min", "current_rate_max_a_per_min",
        "voltage_rate_max_v_per_min",
    )]
    available_contributions = [name for name in contribution_columns if name in summary.columns]
    contributions = record[available_contributions].sort_values(ascending=False).head(3)
    contribution_labels = {
        "deviation_temperature_change_c": "Temperature change",
        "deviation_current_change_a": "Current change",
        "deviation_voltage_std_v": "Voltage variation",
        "deviation_temperature_rate_max_c_per_min": "Temperature rate",
        "deviation_current_rate_max_a_per_min": "Current rate",
        "deviation_voltage_rate_max_v_per_min": "Voltage rate",
    }
    for name, value in contributions.items():
        st.write(f"• **{contribution_labels.get(name, name)}** — robust deviation score {float(value):.2f}")
    st.markdown("**Component measurements**")
    st.dataframe(series, use_container_width=True, hide_index=True)

with trends_tab:
    selected_component = st.selectbox(
        "Component for trend inspection",
        summary["component_id"].astype(str).tolist(),
        key="trend_component",
    )
    selected_parameter = st.selectbox(
        "Measurement",
        ["temperature_c", "current_a", "voltage_v"],
        format_func=lambda v: {
            "temperature_c": "Temperature (°C)",
            "current_a": "Current (A)",
            "voltage_v": "Voltage (V)",
        }[v],
    )
    trend_data = analyzed.loc[
        analyzed["component_id"].astype(str) == selected_component
    ].sort_values("time_min")
    st.line_chart(trend_data.set_index("time_min")[[selected_parameter]])
    st.markdown("**Cross-component feature distribution**")
    feature = st.selectbox(
        "Feature",
        ["temperature_change_c", "current_change_a", "voltage_std_v", "anomaly_score"],
        format_func=lambda v: {
            "temperature_change_c": "Temperature change (°C)",
            "current_change_a": "Current change (A)",
            "voltage_std_v": "Voltage variation (V)",
            "anomaly_score": "Anomaly score",
        }[v],
    )
    st.bar_chart(summary.set_index("component_id")[[feature]].sort_values(feature, ascending=False).head(20))

with export_tab:
    st.subheader("Export results")
    st.write("Download the analyzed component summary or readings with calculated time rates.")
    st.download_button(
        "Download complete screening report",
        data=to_csv_bytes(summary),
        file_name="CODE.KAISEN_screening_report.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download analyzed readings",
        data=to_csv_bytes(analyzed),
        file_name="CODE.KAISEN_analyzed_readings.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download validated input data",
        data=to_csv_bytes(readings),
        file_name="CODE.KAISEN_validated_readings.csv",
        mime="text/csv",
    )
    st.subheader("Data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows received", validation.rows_received)
    q2.metric("Rows analyzed", validation.rows_used)
    q3.metric("Components", validation.components)
    st.write("Required schema")
    st.code(",".join(REQUIRED_COLUMNS), language="text")
    st.caption("Input validation rejects invalid or duplicate component/time readings rather than silently dropping them.")

st.divider()
st.caption(
    "CODE.KAISEN • Prototype screening support • No real-time sensor connection, "
    "persistence, authentication, or validated failure-prediction model is implied by this version."
)
