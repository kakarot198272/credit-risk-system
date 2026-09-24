"""Versioned local artifacts; activation is explicit and atomic."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def model_directory():
    return Path(os.environ.get("CREDIT_RISK_MODEL_DIR", ROOT / "models")).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_path(directory, relative):
    directory = Path(directory).resolve()
    candidate = (directory / relative).resolve()
    if not candidate.is_relative_to(directory):
        raise ValueError("Artifact path must stay inside the model directory")
    return candidate


def active_bundle_path(directory=None):
    directory = Path(directory) if directory is not None else model_directory()
    registry = json.loads((directory / "registry.json").read_text())
    relative = registry.get("production_model")
    if not relative:
        raise FileNotFoundError("No active model. Train and explicitly activate a model first.")
    path = contained_path(directory, relative)
    if not path.is_file():
        raise FileNotFoundError(
            "Active model artifact is missing. Rebuild or restore the recorded artifact."
        )
    return path


def register_candidate(directory, relative_path, *, activate=False):
    directory = Path(directory)
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    path = contained_path(directory, relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    registry.update({"schema_version": 2, "latest_candidate": relative_path})
    if activate:
        registry["previous_model"] = registry.get("production_model")
        registry["production_model"] = relative_path
    write_json(registry_path, registry)
