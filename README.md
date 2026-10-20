# CODE.KAISEN — Component Burn-in Anomaly Screening

An explainable screening-support prototype for **SIH Problem Statement 26170: AI Driven Anomaly Detection in Component Burn-In and Screening**.

## Links

- **Primary demo (Render):** https://anomaly-detection-xxmd.onrender.com/
- **Backup demo (Streamlit Community Cloud):** https://code-kaisen-anamoly-detection.streamlit.app/
- **Source code:** https://github.com/ayushkushwaha020/ANOMALY-DETECTION

Free hosting may sleep or cold-start after inactivity. The public demo has no user authentication; **use synthetic data only and do not upload confidential, customer, or production test data**.

## Current capabilities

- Deterministic synthetic demo dataset and CSV upload.
- Schema and numeric validation; rejects missing/non-finite values, duplicate component/time pairs, empty IDs and insufficient per-component readings.
- Sampling-gap and constant-channel warnings to help identify data-quality problems.
- Per-component trend, spread, peak-to-peak, change and maximum rate features.
- Robust, dataset-relative anomaly screening with configurable percentile cutoff.
- Optional user-supplied lower/upper engineering limits, kept separate from relative statistical flags. Only use limits approved for the specific component and test profile.
- Separate parameter charts with physical units, a review queue, top feature-contribution explanations and CSV exports.
- Reproducibility metadata JSON containing app/engine version, UTC timestamp, input SHA-256 fingerprint, row/component counts and screening settings.
- Regression tests, GitHub Actions CI, Docker support and a model/method card.

## Architecture

```text
Synthetic demo / CSV upload
          |
          v
CSV schema + value validation
          |
          +------> Data-quality warnings
          |
          v
Per-component time-series feature extraction
          |
          v
Robust relative-deviation scoring
          |
          +------> Optional approved absolute-limit checks
          |
          v
Explainable review queue + charts + reproducible exports
```

## CSV format

Required header:

```csv
component_id,time_min,temperature_c,current_a,voltage_v
C0001,0,48.0,0.80,5.00
C0001,1,48.2,0.802,5.00
C0001,2,48.4,0.804,5.01
```

Rules:
- At least two readings per component.
- Non-negative numeric `time_min`.
- At most one row per component/time pair.
- Required numeric values must be finite and present.
- Rows are sorted by component and time before analysis.
- Engine limit: 1,000,000 rows. App upload limit: 25 MB.
- Units and approved engineering limits must match the actual test setup. The app does not infer or certify units from the CSV.

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

GitHub Actions runs tests and compilation checks on Python 3.11 and 3.12 for pushes and pull requests to `main`. Verify the latest workflow run before describing CI as passing.

## Run with Docker

```bash
docker build -t code-kaisen .
docker run --rm -p 8501:8501 code-kaisen
```

Then open http://localhost:8501.

## Deployment and security

The Render service is connected to this repository's `main` branch and auto-deploys commits to that branch. The Streamlit Community Cloud URL is retained as a backup. The included Dockerfile is another deployment option.

The current public demo has no application-level authentication, durable data store, audit service, live sensor/API ingestion, or certified test-equipment integration. Do not use it for confidential or production data. Before private/industrial use, deploy behind an approved identity provider or equivalent access control, enforce HTTPS and network restrictions, define data retention and audit policies, and configure health monitoring, resource limits and rollback procedures. The public Render allow-list is not a substitute for authentication.

## Method and limitations

This is a **statistical anomaly-scoring heuristic**, not a trained or validated failure-prediction model. The engine summarizes component changes, variability, peak-to-peak range and time rates, calculates robust deviations across the current dataset, averages selected deviations and applies a configurable percentile cutoff. Sampling-gap checks are treated as data-quality warnings rather than a component-fault score.

Scores are dataset-relative, are not probabilities and have no measured accuracy claim. The percentile is a triage setting, not a validated pass/fail threshold. The optional engineering-limit check only evaluates limits that a user explicitly supplies; the software cannot determine whether those limits are correct or approved.

The synthetic demo data is for interface demonstration only. No remaining-useful-life estimate, confirmed-defect determination or industrial performance guarantee is provided.

**Safety:** “Potential anomaly” means review required. “No statistical flag” does **not** mean the component passed. Maintain approved screening procedures and require qualified engineering review. Never use this prototype as the sole basis for safety-critical, compliance or production pass/fail decisions.

See [MODEL_CARD.md](MODEL_CARD.md) for the method card and validation requirements.
