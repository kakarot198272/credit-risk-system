# Synthetic demonstration results

**Synthetic data only. These figures demonstrate the evaluation/reporting workflow; they do not establish real borrower performance or business savings.**

The reproducible demo contains 4,000 generated applicants, with 800 in the final test set. Model choice and cutoffs were fixed using validation data before test evaluation.

| Model | ROC-AUC | Average precision | Brier score | Approval rate | Simulated profit (CU) |
|---|---:|---:|---:|---:|---:|
| logistic_baseline | 0.7709 | 0.4089 | 0.1056 | 71.25% | 2,452,249.77 |
| xgboost_shallow | 0.7723 | 0.3923 | 0.1048 | 68.88% | 1,787,269.18 |

Selected on validation: `xgboost_shallow`. Its frozen cutoff is 0.175. Test default prevalence is 14.37%.

**The selected XGBoost policy did not outperform logistic regression on test simulated profit in this demo.** The holdout makes that limitation visible. We did not switch the selected model after seeing the test result or claim a profit improvement. Any further model/policy iteration needs a new validation protocol and an honest final evaluation.

Financial assumptions: margin 10%, loss fraction on default 60%, loan principal as exposure. Actual default labels in this toy dataset are synthetic. Profit remains assumption-based and is not realized revenue.

Verification: 32 automated checks passed on Python 3.12. Checks exercise data isolation, model/serving consistency, financial calculations, explanation consistency, invalid requests, readiness, drift, and dashboard interaction. Docker/Compose execution remains unverified locally because Docker is unavailable; CI defines a container smoke test.

Source fingerprint: `105f82a60406a9c82485c330d18b837be3be64f3d8771df8f4c00805187308c3`.

Reproduce with `python -m scripts.run_demo`. Run IDs and timestamps will change; the seeded synthetic data and modeling protocol are fixed.
