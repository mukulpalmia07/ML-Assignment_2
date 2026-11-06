"""FastAPI app: `uvicorn app.main:app` -> interactive docs at /docs."""
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Breast Cancer Classifier API", description="Five pre-trained classifiers on UCI WDBC features.")
app.include_router(router)
