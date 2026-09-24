# Version 2 implementation notes

## Corrections

- Replaced stressed-data training with original-data training and an explicitly isolated synthetic demo.
- Enforced one row per applicant and removed identifiers from features.
- Added separate training, calibration, validation, and final test subsets.
- Replaced probability-only candidate profit comparison with loan-exposure-weighted simulations using validation outcomes.
- Kept economic assumptions and thresholds with each model version, avoiding an independently drifting API policy.
- Rebuilt inference validation and consistent financial-ratio computation.
- Added SHAP with source-feature grouping and explicit base-log-odds units; preserved sparse representation for XGBoost explanations.
- Fixed PSI edge handling and prevented comparisons across model versions.
- Distinguished already-funded portfolios from future applications during stress scenarios.

## Added

- Streamlit policy explorer, applicant and batch scoring, calibration, model quality, and monitoring views.
- Raw-source feature-building command and an isolated synthetic demonstration generator.
- Artifact manifests, hashes, split assignments, version/dependency metadata, and explicit local activation/rollback.
- Regression tests, lint configuration, GitHub Actions checks, Docker Compose, and a Linux dependency lock.
- Interview guide and a two-minute live-demo/recording walkthrough.

## Migration

The old model registry referenced unavailable pickle files and unverified metrics. The committed registry now has no active model. Generate a new candidate and activate it explicitly. The API accepts a versioned input wrapper (`features` and `explain`) instead of an arbitrary unvalidated dictionary. Fetch `/schema` for the exact input contract.

Legacy exploratory scripts are retained in `experiments/legacy/`. They are excluded from the supported training and CI paths. Their original profit figures and relabeled stress metrics must not be used as validated results.

## Verification limits

The original data and original model artifact were absent from the supplied repository. Development checks therefore use synthetic data. Docker was not available in the workspace; a Compose smoke test is defined in CI. An AWS deployment has not been created or verified.
