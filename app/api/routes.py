"""REST endpoints: score JSON records or a CSV file with all five saved models."""
import json
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config.config import settings
from app.services.data_processor import DataProcessor
from app.services.evaluation_service import Evaluator
from app.services.model_service import ModelService

router = APIRouter()


@lru_cache
def service() -> ModelService:
    return ModelService.load()


def to_json(df: pd.DataFrame, orient: str):
    return json.loads(df.to_json(orient=orient, double_precision=4))  # NaN -> null


def score(df: pd.DataFrame) -> dict:
    try:
        X, y = DataProcessor.split_xy(df, service().features)
    except ValueError as e:
        raise HTTPException(422, str(e))
    probs = service().predict_proba(X)
    return {
        "probability_malignant": to_json(probs, "records"),
        "prediction": to_json(Evaluator.label(probs), "records"),
        "metrics": None if y is None else to_json(Evaluator.compare(y, probs), "index"),
    }


@router.get("/models")
def models() -> dict:
    return {"models": list(service().models), "features": service().features, "target": settings.target}


@router.post("/predict")
def predict(records: list[dict[str, float]]) -> dict:
    """One or more rows as {feature: value}; add `diagnosis` (0/1) to every row to also get metrics."""
    return score(pd.DataFrame(records))


@router.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)) -> dict:
    """Bulk CSV with the 30 feature columns; metrics are included when the `diagnosis` column is present."""
    try:
        df = pd.read_csv(file.file)
    except ValueError as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
    return score(df)
