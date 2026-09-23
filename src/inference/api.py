"""FastAPI inference with explicit readiness and model-specific input validation."""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.inference.predict import InputError, Predictor
from src.risk.artifacts import ROOT

logger = logging.getLogger(__name__)
log_lock = threading.Lock()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]
    explain: bool = True


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicants: list[dict[str, Any]] = Field(min_length=1, max_length=100)
    explain: bool = False


def create_app(model_dir=None, log_path=None):
    prediction_log = Path(
        log_path or os.environ.get("CREDIT_RISK_LOG_PATH", ROOT / "logs/prediction_logs.jsonl")
    )

    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.predictor = Predictor.from_directory(model_dir)
            application.state.load_error = None
        except Exception:
            logger.exception("Model unavailable; /ready will report 503")
            application.state.predictor = None
            application.state.load_error = (
                "No valid active model. Train and activate a model, then restart the API."
            )
        yield

    application = FastAPI(
        title="Credit Risk Decision & Monitoring",
        version="2.0.0",
        description="Portfolio demonstration. Predictions and simulated economics require dataset-specific validation.",
        lifespan=lifespan,
    )

    def predictor(request):
        instance = request.app.state.predictor
        if instance is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error)
        return instance

    def write_logs(results):
        # Store operational outputs only; never record raw applicant features.
        entries = [
            {
                name: row[name]
                for name in [
                    "request_id",
                    "timestamp",
                    "model_version",
                    "dataset_kind",
                    "predicted_pd",
                    "decision",
                    "threshold_used",
                ]
            }
            for row in results
        ]
        try:
            prediction_log.parent.mkdir(parents=True, exist_ok=True)
            with log_lock, prediction_log.open("a") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
        except OSError:
            logger.exception("Could not append prediction audit log")

    @application.get("/")
    @application.get("/health")
    def health(request: Request):
        instance = request.app.state.predictor
        return {
            "status": "running",
            "ready": instance is not None,
            "model_version": instance.version if instance else None,
        }

    @application.get("/ready")
    def ready(request: Request):
        instance = predictor(request)
        return {
            "ready": True,
            "model_version": instance.version,
            "dataset_kind": instance.bundle["dataset_kind"],
        }

    @application.get("/schema")
    def schema(request: Request):
        return predictor(request).metadata()

    @application.post("/predict")
    def predict(data: PredictionRequest, request: Request):
        instance = predictor(request)
        try:
            result = instance.predict(data.features, explain=data.explain)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=500, detail="Prediction failed; check server logs"
            ) from exc
        write_logs([result])
        return result

    @application.post("/predict/batch")
    def batch(data: BatchRequest, request: Request):
        if data.explain and len(data.applicants) > 10:
            raise HTTPException(
                status_code=422, detail="Explanations are limited to 10 applicants per request"
            )
        instance = predictor(request)
        try:
            results = instance.predict_many(data.applicants, explain=data.explain)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Batch prediction failed")
            raise HTTPException(
                status_code=500, detail="Prediction failed; check server logs"
            ) from exc
        write_logs(results)
        return {"model_version": instance.version, "count": len(results), "predictions": results}

    return application


app = create_app()
