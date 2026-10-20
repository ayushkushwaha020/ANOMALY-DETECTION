from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from anomaly_engine import (
    DataValidationError,
    FEATURE_COLUMNS,
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

APP_VERSION = "1.1.0"
ENGINE_VERSION = "1.1.0"
MAX_UPLOAD_MB = 25

# Optional OIDC authentication. Configure Streamlit's [auth] secrets on the
# hosting platform and set require_login=true to gate access to uploaded data.
AUTH_CONFIGURED = "auth" in st.secrets
AUTH_REQUIRED = bool(st.secrets["app"].get("require_login", False)) if "app" in st.secrets else False
if AUTH_REQUIRED and not AUTH_CONFIGURED:
    st.error("Authentication is required but OIDC [auth] secrets are not configured. Ask the deployment owner to configure the identity provider.")
    st.stop()
if AUTH_REQUIRED and not st.user.is_logged_in:
    st.title("CODE.KAISEN — Sign in required")
    st.info("This deployment requires an approved identity-provider login before any data can be accessed.")
    st.button("Sign in", on_click=st.login)
    st.stop()
PARAMETERS = {
    "temperature_c": ("Temperature", "°C"),
    "current_a": ("Current", "A"),
    "voltage_v": ("Voltage", "V"),
}
CONTRIBUTION_LABELS = {
    "deviation_temperature_change_c": "Temperature change",
    "deviation_current_change_a": "Current change",
    "deviation_voltage_std_v": "Voltage variation",
    "deviation_temperature_rate_max_c_per_min": "Temperature rate",
    "deviation_current_rate_max_a_per_min": "Current rate",
    "deviation_voltage_rate_max_v_per_min": "Voltage rate",
    "deviation_temperature_std_c": "Temperature variability",
    "deviation_current_std_a": "Current variability",
    "deviation_voltage_change_v": "Voltage change",
    "deviation_temperature_peak_to_peak_c": "Temperature peak-to-peak",
    "deviation_current_peak_to_peak_a": "Current peak-to-peak",
    "deviation_voltage_peak_to_peak_v": "Voltage peak-to-peak",
    "deviation_sampling_gap_max_min": "Largest sampling gap",
}


@st.cache_data(show_spinner=False)
def demo_data(components: int = 80, points: int = 20, seed: int = 42) -> pd.DataFrame:
    """Repeatable synthetic data for interface demonstrations, not real test data."""
    rng = np.random.default_rng(seed)
    rows = []
    synthetic_outliers = {11, 27, 53, 68}
    for c in range(components):
        component_id = f"C{c + 1:04d}"
        is_outlier = c in synthetic_outliers
        base_temp = rng.normal(48, 2)
        base_current = rng.normal(0.80, 0.05)
        base_voltage = rng.normal(5.0, 0.03)
        for t in range(points):
            temperature = base_temp + 0.18 * t + rng.normal(0, 0.35)
            current = base_current + 0.002 * t + rng.normal(0, 0.008)
            voltage = base_voltage + rng.normal(0, 0.008)
            if is_outlier and t > 10:
                temperature += (t - 10) * 0.55
                current += (t - 10) * 0.010
                voltage += rng.normal(0, 0.035)
            rows.append([component_id, float(t), temperature, current, voltage])
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def parse_optional_number(raw: str, label: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def get_engineering_limits(enabled: bool) -> dict[str, tuple[float, float]]:
    if not enabled:
        return {}
    limits = {}
    for column, (label, unit) in PARAMETERS.items():
        low = parse_optional_number(st.session_state.get(f"limit_low_{column}", ""), f"{label} lower limit")
        high = parse_optional_number(st.session_state.get(f"limit_high_{column}", ""), f"{label} upper limit")
        if low is None or high is None:
            raise ValueError(
                f"Enter both approved {label.lower()} limits before enabling engineering-limit checks."
            )
        if low >= high:
            raise ValueError(f"{label} lower limit must be less than its upper limit.")
        limits[column] = (low, high)
    return limits


def apply_engineering_limits(
    readings: pd.DataFrame, summary: pd.DataFrame, limits: dict[str, tuple[float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add independent absolute-limit results; never convert these into a model pass."""
    frame = readings.copy()
    if not limits:
        summary = summary.copy()
        summary["limit_violation_count"] = 0
        summary["engineering_limit_status"] = "Not configured"
        summary["limit_violation_reasons"] = ""
        return frame, summary

    reason_columns = []
    for column, (low, high) in limits.items():
        label, unit = PARAMETERS[column]
        reason_col = f"limit_violation_{column}"
        frame[reason_col] = (frame[column] < low) | (frame[column] > high)
        reason_columns.append(reason_col)
    frame["engineering_limit_exceeded"] = frame[reason_columns].any(axis=1)
    per_component = frame.groupby("component_id", sort=False).agg(
        limit_violation_count=("engineering_limit_exceeded", "sum")
    ).reset_index()
    reasons = []
    for component_id, group in frame.groupby("component_id", sort=False):
        found = []
        for column, (low, high) in limits.items():
            if ((group[column] < low) | (group[column] > high)).any():
                label, unit = PARAMETERS[column]
                found.append(f"{label} outside [{low:g}, {high:g}] {unit}")
        reasons.append({"component_id": component_id, "limit_violation_reasons": "; ".join(found)})
    per_component = per_component.merge(pd.DataFrame(reasons), on="component_id", how="left")
    per_component["engineering_limit_status"] = np.where(
        per_component["limit_violation_count"] > 0,
        "Approved limit exceeded — review required",
        "No configured limit exceeded — not a pass result",
    )
    summary = summary.drop(
        columns=["limit_violation_count", "engineering_limit_status", "limit_violation_reasons"],
        errors="ignore",
    ).merge(per_component, on="component_id", how="left")
    return frame, summary


def build_report_metadata(
    source_label: str,
    fingerprint: str,
    row_count: int,
    component_count: int,
    percentile: int,
    threshold: float,
    limits: dict[str, tuple[float, float]],
) -> dict:
    return {
        "application": "CODE.KAISEN Component Burn-in Anomaly Screening",
        "app_version": APP_VERSION,
        "engine_version": ENGINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": source_label,
        "input_sha256": fingerprint,
        "rows_analyzed": int(row_count),
        "components_analyzed": int(component_count),
        "screening_percentile": int(percentile),
        "computed_relative_score_cutoff": float(threshold),
        "engineering_limits": {
            key: {"lower": values[0], "upper": values[1], "unit": PARAMETERS[key][1]}
            for key, values in limits.items()
        },
        "method": "Equal-weight average of robust, dataset-relative feature deviations",
        "limitations": [
            "Not a trained or validated failure-prediction model.",
            "Relative score is not a probability, defect rate, or pass/fail decision.",
            "Engineering limits are only checked when explicitly configured by the user.",
            "Synthetic demo data is not evidence of industrial performance.",
        ],
    }


with st.sidebar:
    st.markdown("## ⚡ CODE.KAISEN")
    st.caption("Burn-in & screening intelligence")
    if AUTH_REQUIRED:
        st.caption("Authenticated session")
        st.button("Sign out", on_click=st.logout)
    elif not AUTH_CONFIGURED:
        st.warning("Public demo: do not upload confidential or production data.")
    st.divider()
    source = st.radio("Data source", ["Synthetic demo", "Upload CSV"], index=0)
    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload burn-in readings",
            type=["csv"],
            help="Required columns: " + ", ".join(REQUIRED_COLUMNS),
        )
        st.caption(f"Recommended maximum upload: {MAX_UPLOAD_MB} MB.")
    percentile = st.slider(
        "Relative screening cutoff percentile",
        min_value=80,
        max_value=98,
        value=90,
        step=1,
        help="Higher cutoffs usually flag fewer components. This is not a validated pass/fail threshold.",
    )
    st.divider()
    st.markdown("**Optional engineering limits**")
    limits_enabled = st.checkbox(
        "Check approved absolute limits",
        value=False,
        help="Enable only when you have approved limits for this component and test profile.",
    )
    if limits_enabled:
        st.caption("Enter your approved limits; no default limits are assumed.")
        for column, (label, unit) in PARAMETERS.items():
            st.markdown(f"**{label} ({unit})**")
            lo, hi = st.columns(2)
            lo.text_input("Lower", key=f"limit_low_{column}", placeholder="Approved lower")
            hi.text_input("Upper", key=f"limit_high_{column}", placeholder="Approved upper")
    st.divider()
    st.caption("Method: robust statistical screening across multi-parameter and time-series features.")
    st.caption(f"App {APP_VERSION} · Engine {ENGINE_VERSION}")

st.title("Component Burn-in Anomaly Screening")
st.caption("SIH PS 26170 • Explainable review support • No automated pass/fail decisions")

if source == "Upload CSV":
    if uploaded_file is None:
        st.info("Upload a CSV to begin, or switch to Synthetic demo to explore the dashboard.")
        st.stop()
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File exceeds the app's {MAX_UPLOAD_MB} MB safety limit. Split the dataset and retry.")
        st.stop()
    raw_bytes = uploaded_file.getvalue()
    input_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw_data = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        st.error("Could not read this CSV. Check that it is a valid, non-empty UTF-8 CSV file.")
        st.stop()
    source_label = "CSV upload"
else:
    raw_data = demo_data()
    input_fingerprint = hashlib.sha256(csv_bytes(raw_data)).hexdigest()
    source_label = "synthetic demo"

try:
    limits = get_engineering_limits(limits_enabled)
    readings, validation = validate_readings(raw_data)
    analyzed, summary, threshold = score_anomalies(readings, percentile=float(percentile))
    analyzed, summary = apply_engineering_limits(analyzed, summary, limits)
except DataValidationError as exc:
    st.error(f"**Data validation failed:** {exc}")
    st.info("Correct the CSV headers, values, duplicate readings or component coverage, then retry.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except (TypeError, KeyError, FloatingPointError):
    st.error("The dataset could not be analyzed safely. Check the input format and retry.")
    st.stop()

flagged = summary[summary["status"].str.startswith("Potential anomaly")].copy()
limit_flagged = summary[summary["limit_violation_count"] > 0].copy()

with st.container(border=True):
    st.markdown("#### Screening snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Components analyzed", f"{validation.components:,}")
    m2.metric("Relative flags", f"{len(flagged):,}")
    m3.metric("Limit violations", f"{len(limit_flagged):,}")
    m4.metric("Readings analyzed", f"{len(analyzed):,}")
    st.caption(
        f"Rows validated: {validation.rows_used:,}/{validation.rows_received:,} · "
        f"Relative cutoff: {percentile}th percentile · "
        f"Score threshold: {fmt(threshold)} · "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

for warning in validation.warnings:
    st.warning(warning)

if source == "Synthetic demo":
    st.info(
        "Synthetic demonstration data is being used. It is not real component data and "
        "does not establish industrial accuracy or failure-prediction performance."
    )

st.warning(
    "A statistical flag is a review signal, not a confirmed defect. "
    "No statistical flag does NOT mean a component passed. Do not use this prototype as "
    "the sole basis for safety, compliance or production decisions."
)

overview_tab, flagged_tab, component_tab, trends_tab, export_tab = st.tabs(
    ["Overview", "Review queue", "Component analysis", "Trends & data quality", "Reports"]
)

with overview_tab:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Highest relative anomaly scores")
        view_cols = [
            "component_id", "status", "anomaly_score", "temperature_change_c",
            "current_change_a", "voltage_std_v", "limit_violation_count",
        ]
        st.dataframe(
            summary[view_cols].head(15).rename(columns={
                "component_id": "Component ID",
                "status": "Statistical screening result",
                "anomaly_score": "Relative score",
                "temperature_change_c": "Temperature change (°C)",
                "current_change_a": "Current change (A)",
                "voltage_std_v": "Voltage variation (V)",
                "limit_violation_count": "Out-of-limit readings",
            }),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Relative screening results")
        counts = summary["status"].str.startswith("Potential anomaly").value_counts()
        chart_data = pd.DataFrame({
            "Result": ["Potential anomaly — review", "No statistical flag — not a pass"],
            "Components": [
                int(summary["status"].str.startswith("Potential anomaly").sum()),
                int((~summary["status"].str.startswith("Potential anomaly")).sum()),
            ],
        }).set_index("Result")
        st.bar_chart(chart_data)
        st.caption("Counts depend on this dataset and the selected percentile.")
    st.subheader("Absolute-limit review")
    if not limits:
        st.info("Absolute engineering limits are not configured. Only relative statistical screening is active.")
    elif limit_flagged.empty:
        st.success("No configured limit was exceeded in these readings. This is not a production pass result.")
    else:
        st.dataframe(
            limit_flagged[["component_id", "limit_violation_count", "limit_violation_reasons"]],
            width="stretch",
            hide_index=True,
        )

with flagged_tab:
    st.subheader("Components requiring review")
    st.caption("Statistical flags and configured engineering-limit violations are separate signals.")
    if flagged.empty and limit_flagged.empty:
        st.info("No components were flagged by the current settings. This does not prove all components are healthy.")
    else:
        review_ids = sorted(set(flagged["component_id"]) | set(limit_flagged["component_id"]))
        review = summary[summary["component_id"].isin(review_ids)].copy()
        display_columns = [
            "component_id", "status", "anomaly_score", "limit_violation_count",
            "limit_violation_reasons", "temperature_change_c", "current_change_a",
            "voltage_std_v", "temperature_rate_max_c_per_min",
            "current_rate_max_a_per_min", "voltage_rate_max_v_per_min",
        ]
        review = review[display_columns].rename(columns={
            "component_id": "Component ID",
            "status": "Statistical result",
            "anomaly_score": "Relative score",
            "limit_violation_count": "Out-of-limit readings",
            "limit_violation_reasons": "Limit reasons",
            "temperature_change_c": "Temperature change (°C)",
            "current_change_a": "Current change (A)",
            "voltage_std_v": "Voltage variation (V)",
            "temperature_rate_max_c_per_min": "Max |temperature rate| (°C/min)",
            "current_rate_max_a_per_min": "Max |current rate| (A/min)",
            "voltage_rate_max_v_per_min": "Max |voltage rate| (V/min)",
        })
        st.dataframe(review, width="stretch", hide_index=True)
        st.download_button(
            "Download review queue CSV",
            data=csv_bytes(review),
            file_name="CODE_KAISEN_review_queue.csv",
            mime="text/csv",
        )

with component_tab:
    component_ids = summary["component_id"].astype(str).tolist()
    selected_id = st.selectbox("Choose a component", component_ids)
    series = analyzed[analyzed["component_id"].astype(str) == selected_id].sort_values("time_min")
    record = summary[summary["component_id"].astype(str) == selected_id].iloc[0]
    x, y, z = st.columns(3)
    x.metric("Statistical result", str(record["status"]))
    y.metric("Relative score", fmt(record["anomaly_score"]))
    z.metric("Out-of-limit readings", int(record["limit_violation_count"]))
    st.markdown("**Temperature**")
    st.line_chart(series.set_index("time_min")[["temperature_c"]])
    st.markdown("**Current**")
    st.line_chart(series.set_index("time_min")[["current_a"]])
    st.markdown("**Voltage**")
    st.line_chart(series.set_index("time_min")[["voltage_v"]])
    st.caption("Each parameter is plotted separately so different physical units do not obscure one another.")
    st.markdown("**Why was this component flagged?**")
    contribution_columns = [f"deviation_{name}" for name in FEATURE_COLUMNS]
    available = [name for name in contribution_columns if name in summary.columns]
    contributors = record[available].sort_values(ascending=False).head(5)
    if float(record["anomaly_score"]) == 0:
        st.write("No relative feature deviation was detected by this scoring method.")
    else:
        for name, value in contributors.items():
            st.write(f"• **{CONTRIBUTION_LABELS.get(name, name)}** — relative deviation contribution {float(value):.2f}")
    if limits:
        st.markdown("**Approved engineering-limit check**")
        st.write(record["engineering_limit_status"])
        if record["limit_violation_reasons"]:
            st.write(record["limit_violation_reasons"])
    st.markdown("**Measurements and calculated rates**")
    st.dataframe(series, width="stretch", hide_index=True)

with trends_tab:
    st.subheader("Trend inspection")
    selected_component = st.selectbox(
        "Component",
        summary["component_id"].astype(str).tolist(),
        key="trend_component",
    )
    selected_parameter = st.selectbox(
        "Measurement",
        list(PARAMETERS),
        format_func=lambda v: f"{PARAMETERS[v][0]} ({PARAMETERS[v][1]})",
    )
    trend_data = analyzed[analyzed["component_id"].astype(str) == selected_component].sort_values("time_min")
    st.line_chart(trend_data.set_index("time_min")[[selected_parameter]])
    st.subheader("Cross-component feature comparison")
    feature_options = [
        "temperature_change_c", "current_change_a", "voltage_std_v",
        "temperature_std_c", "current_std_a", "voltage_change_v",
        "temperature_rate_max_c_per_min", "current_rate_max_a_per_min",
        "voltage_rate_max_v_per_min", "sampling_gap_max_min", "anomaly_score",
    ]
    selected_feature = st.selectbox("Feature", feature_options)
    st.bar_chart(summary.set_index("component_id")[[selected_feature]].sort_values(selected_feature, ascending=False).head(20))
    st.subheader("Data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows received", validation.rows_received)
    q2.metric("Rows analyzed", validation.rows_used)
    q3.metric("Components", validation.components)
    st.write("Required CSV schema")
    st.code(",".join(REQUIRED_COLUMNS), language="text")
    st.caption("Missing/non-finite measurements and duplicate component/time pairs are rejected, not silently dropped.")

with export_tab:
    st.subheader("Download screening results")
    st.caption("Exports are analysis aids, not certificates of component health or production acceptance.")
    st.download_button(
        "Download full component summary CSV",
        data=csv_bytes(summary),
        file_name="CODE_KAISEN_screening_report.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download analyzed readings CSV",
        data=csv_bytes(analyzed),
        file_name="CODE_KAISEN_analyzed_readings.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download validated input CSV",
        data=csv_bytes(readings),
        file_name="CODE_KAISEN_validated_readings.csv",
        mime="text/csv",
    )
    metadata = build_report_metadata(
        source_label, input_fingerprint, len(analyzed), validation.components,
        percentile, threshold, limits,
    )
    st.download_button(
        "Download run metadata JSON",
        data=json.dumps(metadata, indent=2, sort_keys=True),
        file_name="CODE_KAISEN_run_metadata.json",
        mime="application/json",
    )
    st.markdown("**Reproducibility metadata**")
    st.code(
        f"Input SHA-256: {input_fingerprint}\n"
        f"App version: {APP_VERSION}\nEngine version: {ENGINE_VERSION}\n"
        f"UTC generated: {metadata['generated_at_utc']}",
        language="text",
    )
    st.caption("The input fingerprint identifies the exact source bytes; it does not prove data authenticity.")

st.divider()
st.caption(
    "CODE.KAISEN • Screening-support prototype • No authentication, durable storage, audit service, "
    "live sensor ingestion or validated failure-prediction model is implied."
)
