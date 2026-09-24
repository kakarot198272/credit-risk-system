# Two-minute demonstration walkthrough

Start the API and dashboard using the README. Keep the synthetic-data banner visible. This is a walkthrough for a live demo or a recording; it is not a claim that the system has processed real loan applications.

| Time | Action | Explain |
|---|---|---|
| 0:00–0:20 | Open Policy explorer | The project links risk estimates to lending decisions. Identify the validation population and financial assumptions. |
| 0:20–0:45 | Change the approval cutoff | Show approval rate, default rate among approved borrowers, and outcome-based simulated profit. Explain the trade-off. |
| 0:45–1:05 | Expand the stress scenario | Already-funded loans cannot be removed from the portfolio. New applications can be reassessed. The shock is hypothetical. |
| 1:05–1:30 | Open Applicant scoring and score the example | Show PD, saved-policy recommendation, and SHAP factors. Explain that the factors use base-model log odds. |
| 1:30–1:50 | Open Model quality | Show the frozen final test, logistic baseline, and calibration chart. Explain the separate train/calibration/validation/test roles. |
| 1:50–2:00 | Open Project evidence | Show versioning and report download; state the remaining real-data validation requirement. |

The dashboard's scenario controls never mutate the active API policy. Use the supplied CSV template for batch scoring. Screenshots or recordings should retain the synthetic-demo label until a separate real-data run is available.

For an API-only smoke check:

```python
import requests
base = "http://127.0.0.1:8000"
schema = requests.get(base + "/schema", timeout=10).json()
response = requests.post(base + "/predict", json=schema["example_request"], timeout=60)
response.raise_for_status()
print(response.json())
```
