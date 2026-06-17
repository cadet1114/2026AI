# Agent Handoff

Last updated: 2026-06-16

## Status

P0/P1 upgrade is implemented on `feature/p0-p1-bayesian-demo`:

- Full 11-node discrete Bayesian network with exact inference and evidence attribution.
- Expert CPT baseline and learned CPT variant with the same graph structure.
- Official USGS download, checksum metadata, hybrid-data provenance, five-fold evaluation, calibration and missing-evidence tests.
- Coordinate-aware risk A*, independent ground/air graphs, constrained assignment and stateful rescue simulation.
- Resource-aware expected utility with six-term contribution breakdowns, candidate audit matrices and deterministic explanations.
- Runtime JSON Schema validation at both pipeline boundaries.
- Streamlit visual demo with risk-layer map, event controls, utility decision console, evidence, metrics and downloadable output.

## Verification

- `40 passed` with `.venv/bin/pytest -q`.
- Full experiment: 50,000 samples, 5 folds, 92.566 seconds, 121.632 MB peak traced memory on M3 / 16GB.
- Learned CPT improves trapped F1 from 0.3354 to 0.4041 and road ROC-AUC from 0.7740 to 0.7766.
- Fixed and learned v2 outputs both pass `schemas/decision_output.schema.json`.

## Important Files

- `app.py`: visual demo entry point.
- `src/emergency_commander/bayesian_network.py`: exact BN inference and CPT fitting.
- `src/emergency_commander/experiment.py`: hybrid-data and cross-validation pipeline.
- `src/emergency_commander/pipeline.py`: stateful end-to-end decision pipeline.
- `artifacts/full_bayesian_experiment/experiment_report.md`: experiment summary.
- `examples/decision_output_*_v2.json`: synchronized demo outputs.

## Reproduction

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/streamlit run app.py
```

The generated `hybrid_dataset.jsonl` is intentionally ignored because it is reproducible and about 27MB. Public source rows, metadata, learned CPT, metrics, runtime and report remain versioned.
