# CODE.KAISEN — Component Burn-in Anomaly Detection

An explainable screening-support prototype for **SIH Problem Statement 26170: AI Driven Anomaly Detection in Component Burn-In and Screening**.

## Links

- **Live demo:** https://code-kaisen-anamoly-detection.streamlit.app/
- **GitHub:** https://github.com/ayushkushwaha020/ANOMALY-DETECTION

## What it does

- Starts with deterministic synthetic data or accepts a burn-in CSV upload.
- Validates schema, numeric readings, component identifiers, timestamps, duplicate readings, and minimum readings per component.
- Extracts per-component temperature/current/voltage changes and time rates.
- Produces a robust, relative anomaly score and configurable percentile cutoff.
- Shows overview, flagged components, component trends, data quality, and CSV exports.
- Labels flags as **potential anomalies for engineer review**, not confirmed defects.

## Architecture

```text
Synthetic demo / CSV upload
          |
          v
Schema + value validation
          |
          v
Per-component feature extraction
          |
          v
Robust statistical deviation score
          |
          v
Configurable percentile triage cutoff
          |
          v
Streamlit dashboard + CSV exports
```

## CSV format

Required header:

```csv
component_id,time_min,temperature_c,current_a,voltage_v
C0001,0,48.0,0.80,5.00
C0001,1,48.2,0.802,5.00
C0001,2,48.4,0.804,5.01
```

Requirements:
- At least two readings per component.
- Non-negative numeric `time_min`.
- At most one row per component/time pair.
- All required numeric measurements must be finite and present.
- Rows are sorted by component and time before analysis.
- Maximum accepted dataset size: 1,000,000 rows in the engine; upload size is also limited by deployment configuration.

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app.py anomaly_engine.py
```

GitHub Actions runs tests and compilation checks for Python 3.11 and 3.12 on pushes and pull requests to `main`.

## Run with Docker

```bash
docker build -t code-kaisen .
docker run --rm -p 8501:8501 code-kaisen
```

Then open http://localhost:8501.

## Deployment notes

The existing Streamlit Community Cloud app is connected to this repository's `main` branch and `app.py` entrypoint. Commits to `main` trigger redeployment while the configured app URL remains the same. The app may sleep on free Community Cloud when inactive; code changes do not remove host-level sleep limits.

The included Dockerfile is an additional deployment option for a container host. A production host should provide HTTPS, resource limits, health monitoring, controlled access, and a defined update/rollback process.

## Method and limitations

This is a **statistical anomaly-scoring heuristic**, not a trained or validated failure-prediction model. Scores are relative to the current dataset, are not probabilities, and have no accuracy claim. The percentile setting controls triage volume; it is not a validated pass/fail threshold.

The current app does not provide user authentication, persistent data storage, an audit trail, live sensor/API ingestion, or certified integration with test equipment. Synthetic demo readings are not industrial evidence. See [MODEL_CARD.md](MODEL_CARD.md) before interpreting results.

**Safety:** An anomaly flag does not establish that a component is defective. Do not use this prototype as the sole basis for safety-critical, compliance, or production pass/fail decisions. Maintain approved screening limits and require qualified engineering review.
