import pytest

from scripts.make_demo import generate_demo
from src.inference.predict import Predictor
from src.models.train import train_experiment


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("credit-risk")
    data = root / "synthetic.parquet"
    frame = generate_demo(1200, seed=2026)
    frame.to_parquet(data, index=False)
    directory = root / "models"
    run = train_experiment(data, directory, dataset_kind="synthetic", activate=True, quick=True)
    return {
        "root": root,
        "data": data,
        "frame": frame,
        "directory": directory,
        "run": run,
        "predictor": Predictor.from_directory(directory),
    }


@pytest.fixture
def applicant(trained_run):
    return trained_run["predictor"].metadata()["example_request"]["features"].copy()
