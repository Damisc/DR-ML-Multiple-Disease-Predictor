from fastapi import FastAPI

from src.backend.api.routes import router

app = FastAPI(
    title="DR. ML Prediction App",
    version="1.0.0",
    description="Multi-Disease prediction API"
)

app.include_router(router, prefix="/api")