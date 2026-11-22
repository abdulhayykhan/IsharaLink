"""
main.py -- FastAPI backend service for IsharaLink Pakistan Sign Language recognition.

Responsible for:
  - Serving GET /health (model status, classes count)
  - Serving GET /classes (supported vocabulary list)
  - Serving POST /predict (raw browser landmarks -> preprocessing -> inference)
  - CORS middleware for browser clients
  - Latency measurement without logging sensitive landmark data
"""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import sys
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path so imports work regardless of launch location
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas import (
    ClassesResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.preprocessing.preprocess import preprocess_window
from src.models.predict import load_assets, predict_window

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("isharalink.backend")

# Global state loaded once at startup
APP_STATE = {
    "model_loaded": False,
    "classes": [],
    "threshold": 0.30,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, classes and config once at startup, never per request."""
    logger.info("Initializing IsharaLink backend...")
    try:
        load_assets()
        APP_STATE["model_loaded"] = True
        
        # Load classes list
        classes_path = ROOT_DIR / "data" / "processed" / "classes.json"
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                APP_STATE["classes"] = json.load(f)
        
        # Load config threshold
        config_path = ROOT_DIR / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                APP_STATE["threshold"] = float(cfg.get("threshold", 0.30))
                
        logger.info(
            "Assets loaded successfully. Classes count: %d, Threshold: %.2f",
            len(APP_STATE["classes"]),
            APP_STATE["threshold"]
        )
    except Exception as exc:
        logger.error("Failed to load model assets at startup: %s", exc, exc_info=True)
        APP_STATE["model_loaded"] = False
        
    yield
    logger.info("Shutting down IsharaLink backend...")


app = FastAPI(
    title="IsharaLink API",
    description="Real-time Pakistan Sign Language Recognition API (FastAPI + GRU)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure total request processing time and attach X-Process-Time-Ms header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors and format into a readable error message."""
    errors = exc.errors()
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, errors)
    messages = [f"{e.get('msg', 'Invalid input')} (at {'.'.join(str(x) for x in e.get('loc', []))})" for e in errors]
    return JSONResponse(
        status_code=422,
        content={"detail": " | ".join(messages) if messages else "Invalid request payload"}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check and model loading verification."""
    return HealthResponse(
        status="ok",
        model_loaded=APP_STATE["model_loaded"],
        classes_count=len(APP_STATE["classes"]),
        threshold=APP_STATE["threshold"]
    )


@app.get("/classes", response_model=list[str])
async def get_classes():
    """Returns the ordered list of 62 supported PSL vocabulary words matching classes.json."""
    if not APP_STATE["classes"]:
        # Fallback reload if accessed before lifecycle or in testing
        classes_path = ROOT_DIR / "data" / "processed" / "classes.json"
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                APP_STATE["classes"] = json.load(f)
    return APP_STATE["classes"]


@app.post("/predict", response_model=PredictResponse)
async def predict_sign(payload: PredictRequest):
    """
    Accepts raw MediaPipe hand landmarks from browser, preprocesses into (60, 126),
    and predicts the PSL sign using the trained GRU sequence model.
    """
    if not APP_STATE["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please verify server startup logs."
        )

    t0 = time.perf_counter()

    # 1. Convert validated frames into raw list of dicts expected by preprocess_window
    raw_frames = [{"left": f.left, "right": f.right} for f in payload.frames]

    # 2. Run preprocessing: assemble -> resample to 60 -> wrist-relative normalization
    try:
        arr_preprocessed = preprocess_window(raw_frames)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(exc)}")

    # 3. Model inference: GRU prediction + confidence thresholding
    try:
        result = predict_window(arr_preprocessed)
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    inference_ms = (time.perf_counter() - t0) * 1000.0

    # Privacy by design: NEVER log raw landmark coordinates. Log only timing & outcome.
    logger.info(
        "Predict: frames_in=%d -> label='%s' confidence=%.4f status='%s' latency=%.2fms",
        len(payload.frames),
        result["label"],
        result["confidence"],
        result["status"],
        inference_ms
    )

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        top3=result["top3"],
        status=result["status"]
    )


# Mount static assets (results charts, golden test cases, and frontend application)
from fastapi.staticfiles import StaticFiles

results_dir = ROOT_DIR / "results"
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")

tests_dir = ROOT_DIR / "tests"
if tests_dir.exists():
    app.mount("/tests", StaticFiles(directory=str(tests_dir)), name="tests")

frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

