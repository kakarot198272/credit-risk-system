"""Report saved validation calibration diagnostics for all candidates."""

import json

from src.risk.artifacts import active_bundle_path


def main():
    report = json.loads((active_bundle_path().parent / "report.json").read_text())
    for name, result in report["validation"].items():
        print(
            name,
            json.dumps(
                {"metrics": result["metrics"], "calibration": result["calibration"]}, indent=2
            ),
        )


if __name__ == "__main__":
    main()
