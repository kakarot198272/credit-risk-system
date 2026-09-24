"""Explicit local activation or rollback. Restart API/dashboard after switching."""

import argparse
import json
from pathlib import Path

from src.inference.predict import Predictor
from src.risk.artifacts import contained_path, model_directory, register_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_directory())
    parser.add_argument(
        "--artifact", required=True, help="Relative path to a trusted run's bundle.joblib"
    )
    parser.add_argument("--activate", action="store_true", required=True)
    args = parser.parse_args()
    path = contained_path(args.model_dir, args.artifact)
    predictor = Predictor(path)
    register_candidate(args.model_dir, args.artifact, activate=True)
    print(
        json.dumps(
            {
                "activated": predictor.version,
                "dataset_kind": predictor.bundle["dataset_kind"],
                "next_step": "Restart API/dashboard to load this version",
            }
        )
    )


if __name__ == "__main__":
    main()
