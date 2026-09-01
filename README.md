# Supplementary material — CVC 2027 submission

Anonymous supplementary material for the double-blind submission
*Structured Noise in Diffusion Models: An Audited Survey with a
Finite-Dimensional Protocol for Calibrated Error Propagation*.

- `S1/` — Supplement S1: the anonymised row-level audit log and the
  exact search record of the survey corpus (cited in Section 3 of the
  paper, Review scope and methodology).
- `configurations_and_results/` — the released configurations and
  machine-readable results of the two experiments (Section 5):
  - `m4raw/`: the full M4Raw pipeline (`experience_m4raw_v4.py`; the
    frozen configuration — splits, routes, criteria, controls, triads,
    resampling design — is embedded in the script and echoed in the
    results file) and the machine-readable verdicts
    (`m4raw_v4_results.json`).
  - `brainweb/`: the BrainWeb propagation experiment pipeline
    (`experience2_propagation.py`) and its machine-readable results
    (`exp2_results.json`).

Absolute filesystem paths recorded in the result files have been
redacted (`/home/anon/...`) for double-blind anonymity; no numerical
content was modified.
