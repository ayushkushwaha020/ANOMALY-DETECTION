# CODE.KAISEN anomaly screening — model/method card

## Intended use
A prototype decision-support tool for reviewing electronic component burn-in readings. It highlights components whose statistical features differ from other components in the current dataset and can optionally check user-supplied engineering limits.

## Method
This is an explainable statistical heuristic, not a trained supervised model. It extracts per-component changes, variability, peak-to-peak ranges, maximum absolute rates and sampling-gap summaries for temperature, current and voltage. Robust deviations use median absolute deviation with a standard-deviation fallback; selected measurement features are averaged into a relative score and compared with a configurable percentile.

Sampling gaps and constant channels are reported as data-quality warnings; sampling-gap size is not included in the component fault score. Optional lower/upper engineering limits are an independent rule-based check and must be approved for the exact component and test profile by a qualified engineer.

## Inputs
Required CSV columns: `component_id`, `time_min`, `temperature_c`, `current_a`, `voltage_v`. Each component needs at least two readings; times must be non-negative and unique per component. Missing, non-numeric, infinite or duplicate component/time values are rejected. The app accepts files up to 25 MB; the engine has a 1,000,000-row safety limit.

## Outputs
A dataset-relative anomaly score, a review-oriented status, feature summaries, optional absolute-limit violation counts, charts, CSV reports and a JSON run metadata file. Metadata includes app/engine version, timestamp, source type, input SHA-256 fingerprint, row/component counts, percentile, score threshold and configured limits.

## Known limitations
- No representative labelled ground-truth dataset or measured precision/recall is bundled.
- The percentile is a triage setting, not a validated decision threshold or probability.
- Relative scoring can miss a batch-wide shift if most components share the same abnormal behaviour.
- Feature averaging uses a heuristic equal-weight design and has not been validated against domain acceptance criteria.
- Small, unrepresentative datasets can make rankings unstable.
- Different component families, test recipes, ambient/chamber conditions, sensor calibration and sampling intervals change normal behaviour.
- CSV column names do not verify physical units or data provenance.
- The app does not estimate remaining useful life, confirm defects or prove a component has passed.
- Synthetic demo data is only for demonstrating the interface.
- The public demo does not provide app authentication, durable storage, an audit service, live sensor/API ingestion or certified test-equipment integration.
- A SHA-256 fingerprint identifies input bytes but does not establish source authenticity or chain of custody.

## Safe use
Treat each flag as a prompt for qualified engineering review. “No statistical flag” is not a pass result. Keep approved test limits and safety procedures in force. Do not use this prototype as the sole pass/fail gate for production, safety-critical or compliance decisions. Do not upload confidential or production data to the public demo.

## Validation required before industrial use
1. Define a narrow component family and burn-in test profile with domain engineers.
2. Collect representative, de-identified historical runs with known outcomes and documented test conditions.
3. Define ground truth, acceptable false-alarm/missed-failure rates and decision costs before tuning.
4. Split evaluation by time, production lot or component batch to reduce leakage.
5. Compare the current heuristic with simple engineering rules and suitable ML baselines.
6. Report precision, recall, false-positive rate, missed-anomaly rate, calibration where relevant and uncertainty intervals.
7. Test sensor faults, missing intervals, batch-wide shifts, new component families and distribution drift.
8. Run in shadow mode beside the approved process and have qualified engineers review disagreements.
9. Complete security, privacy, audit, availability and change-control reviews before operational use.
