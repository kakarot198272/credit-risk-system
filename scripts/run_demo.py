"""Rebuild a deterministic, isolated synthetic demonstration."""

import argparse
from pathlib import Path

from scripts.make_demo import generate_demo
from src.models.train import train_experiment
from src.risk.artifacts import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/demo")
    parser.add_argument("--rows", type=int, default=4000)
    args = parser.parse_args()
    data = args.output_root / "data/model_input.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    generate_demo(args.rows).to_parquet(data, index=False)
    run = train_experiment(
        data, args.output_root / "models", dataset_kind="synthetic", activate=True, quick=True
    )
    write_json(
        args.output_root / "demo_manifest.json",
        {
            "dataset_kind": "synthetic",
            "run_directory": str(run),
            "purpose": "Software demonstration; not real credit performance",
        },
    )
    print(
        "Demo is ready. Start the API and dashboard with CREDIT_RISK_MODEL_DIR pointing to artifacts/demo/models."
    )


if __name__ == "__main__":
    main()
