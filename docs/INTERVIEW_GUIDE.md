# Understand and explain this project

Work through these questions with the code open. Understand the calculations and trade-offs before rehearsing an answer.

## 1. What problem does it address?

A lender needs an estimate of an applicant's default risk and a policy for acting on it. The model supplies a probability. A separate policy decides whether to approve under simplified economic assumptions. Monitoring examines whether the distribution of incoming predictions changes.

The current working demonstration uses synthetic applicants. The repository also implements a path for the original Home Credit-shaped dataset; its real-data results are not yet verified.

## 2. What is one row?

One applicant, identified during data preparation by `SK_ID_CURR`. The target `TARGET=1` indicates a recorded default label in the supplied data. Confirm the source dataset's exact target definition and observation horizon before explaining real results. Multiple bureau and previous-application records are aggregated into one applicant-level summary.

Read `src/features/build_features.py`. Be able to explain why directly joining multiple child-table records would multiply rows, why a left join retains applicants without history, and why a missing history amount is different from a known zero.

## 3. What are the useful features?

Income, requested credit, annuity, bureau borrowing summaries, and previous-application summaries. Ratios connect these quantities: for example, requested credit divided by income describes borrowing relative to income. Undefined ratios remain missing and are imputed using training data. The API recomputes ratios using the same function.

Read `src/models/data.py`. Identifiers, synthetic markers, and target labels are excluded from features.

## 4. Why four data subsets?

- Training fits preprocessing and classifiers.
- Calibration adjusts probabilities using separate labels while the base model is frozen.
- Validation chooses the candidate and lending cutoff.
- Final test assesses the already-selected model and cutoff.

Read `src/models/train.py` and `tests/test_training.py`. One regression test reverses only test labels and verifies that model selection and validation results stay unchanged. Explain why tuning on the test result would make it cease to be an honest holdout.

## 5. Why logistic regression and XGBoost?

Logistic regression gives a simple baseline. XGBoost can represent nonlinear relationships and interactions. The experiment compares two fixed tree configurations and their calibrated variants on the same applicant splits. It is deliberately a small candidate search, not a claim of exhaustive model optimization.

The selected model may be raw XGBoost, calibrated XGBoost, or the baseline. A calibration step does not guarantee a better result; inspect its probability metrics and validation outcomes.

## 6. What do the metrics mean?

- ROC-AUC: ranking defaulted applicants above non-defaulted applicants across cutoffs; it is not accuracy.
- Average precision: precision/recall-oriented ranking summary; compare it with default prevalence.
- Brier score: average squared probability error, reflecting overall probability quality.
- Log loss: penalizes wrong confident probability predictions.
- Calibration curve: among applicants with similar predicted risk, compare that risk with the observed default frequency.

A model can rank borrowers well and still assign poor probability values. Probability quality matters because financial expectations use those values directly.

## 7. How does the financial objective work?

Read `src/risk/policy.py`. With amount `A`, margin `m`, loss fraction `L`, probability `p`, and observed default label `y`:

- Expected profit if funded: `A * ((1-p)*m - p*L)`.
- Outcome-based simulated profit: `A * ((1-y)*m - y*L)`.

Example: a repaid 100-unit loan at 10% margin contributes +10 units. A defaulted 1,000-unit loan at 60% loss contributes -600 units. The combined simulated profit is -590, even if the model had predicted both were safe.

This is why candidate selection uses observed validation outcomes. It still assumes simplified margins and recoveries; do not call it realized revenue or a proven return on lending capital.

## 8. What happens during an API request?

FastAPI validates the request wrapper. The Predictor validates every feature against the artifact's schema, recomputes financial ratios, applies the fitted preprocessing/model, and compares PD with the versioned policy cutoff. It returns a decision recommendation and explanation, then logs operational metadata. The model is loaded once at startup, not retrained for each request.

Read `src/inference/api.py`, `src/inference/predict.py`, and the API regression tests.

## 9. What does SHAP explain?

The factors describe contributions to the base model's log odds, with one-hot category contributions grouped back to the original feature. They do not establish causal effects. For calibrated variants, the final probability has an additional calibration transformation and the SHAP contributions do not add directly to that probability.

A meaningful engineering bug caught by the tests: XGBoost treats implicit sparse entries differently from explicit dense zeros. Densifying inputs before explanation produced factors for a different numerical input. The corrected explainer preserves the representation used for scoring, and an additivity check verifies consistency.

## 10. What is monitored?

Prediction-score PSI compares a baseline score distribution with logged scores from the exact same model version. The bins include extreme values and handle constant baseline distributions. A changed score distribution is a prompt for investigation, not proof of degraded model accuracy. Later outcomes are needed to measure prediction quality in production.

## 11. What can you honestly claim now?

You can describe the implemented pipeline, strict API, policy simulator, explanations, versioning, synthetic demonstration, and the checks actually executed. Explain your original work and the improvements you made with assistance accurately. Do not claim real borrowers served, dollars saved, AWS deployment, real-data AUC, regulatory compliance, or experiment-driven business lift without supporting evidence.

## 12. What remains to strengthen the study?

Run the original data through the new pipeline, verify feature availability at application time, inspect calibration and subgroup errors, and assess temporal generalization if suitable timestamps are available. Add delayed-outcome monitoring and a suitable deployment/access-control design before any external pilot.
