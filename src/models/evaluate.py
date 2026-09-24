"""Display the frozen final report without fitting or selecting on test data."""

import argparse
from pathlib import Path

from src.risk.artifacts import active_bundle_path, model_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_directory())
    args = parser.parse_args()
    path = active_bundle_path(args.model_dir).parent / "report.json"
    print(path.read_text())


if __name__ == "__main__":
    main()
