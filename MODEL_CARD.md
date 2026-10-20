# CODE.KAISEN anomaly screening — model/method card

## Intended use
A prototype decision-support tool for exploring electronic component burn-in readings. It highlights components whose statistical features are unusual relative to the other components in the current dataset.

## Method
The current implementation is an explainable statistical heuristic, not a trained supervised model. It extracts per-component changes and rates for temperature, current, and voltage; calculates robust deviations using median absolute deviation (with a standard-deviation fallback); averages selected feature deviations; and applies a configurable percentile cutoff.

## Inputs
CSV columns: `component_id`, `time_min`, `temperature_c`, `current_a`, `voltage_v`. Each component must have at least two readings. Times must be unique per component and non-negative. Missing, non-numeric, infinite, or duplicate component/time values are rejected.

## Outputs
A relative anomaly score, a configurable triage status, feature summaries, and downloadable CSV reports. Scores are dataset-relative and are not probabilities.

## Known limitations
- No validated ground-truth dataset or measured precision/recall is bundled.
- The percentile is a review-volume setting, not a validated decision threshold.
- Relative scoring can be unstable on small or unrepresentative datasets.
- Different test profiles, hardware families, operating conditions, and sampling intervals can change normal behavior.
- This implementation does not estimate remaining useful life or prove component failure.
- The synthetic demo data is for interface demonstration only.
- The app does not currently provide authentication, persistent storage, audit logging, a sensor gateway, or a certified industrial integration.

## Safe use
Treat every flag as a prompt for qualified engineering review. Keep existing approved test limits and safety procedures. Do not use the prototype as the sole pass/fail gate for production, safety-critical, or compliance decisions.

## Validation needed before industrial use
Obtain representative, labeled historical data; define acceptance criteria with domain engineers; evaluate by component lot and test condition; report precision, recall, false-positive rate, and missed-anomaly rate; test data drift and sensor faults; and conduct a controlled shadow deployment before operational reliance.
