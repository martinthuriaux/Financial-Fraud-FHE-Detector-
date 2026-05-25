"""
server.py
---------
FastAPI service that loads a trained Concrete-ML model, compiles it to
an FHE circuit at startup, and exposes a /predict endpoint for encrypted
fraud detection inference.

The model to serve is chosen via the MODEL environment variable:
    MODEL=xgboost         python server.py
    MODEL=decision_tree   python server.py
    MODEL=nn              python server.py

Dependencies
------------
- preprocessing.py must have run (produces data/splits/ and scaler.pkl)
- The chosen model script must have run (produces artifacts/<model>/)
- Each model script saves artifacts/<model>/cml_model.json BEFORE compile()
"""

from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from concrete.ml.common.serialization.loaders import load as cml_load


# ---------------------------------------------------------------------------
# Section 1 — Configuration
# ---------------------------------------------------------------------------

# Which model to serve.  Defaults to xgboost if not set.
MODEL_NAME    = os.environ.get("MODEL", "xgboost")
VALID_MODELS  = {"xgboost", "decision_tree", "nn"}

if MODEL_NAME not in VALID_MODELS:
    raise ValueError(
        f"Invalid MODEL env var: {MODEL_NAME!r}. "
        f"Must be one of {VALID_MODELS}."
    )

# Paths
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts" / MODEL_NAME


# ---------------------------------------------------------------------------
# Section 2 — Pydantic models for request/response validation
# ---------------------------------------------------------------------------
# FastAPI uses these classes to automatically:
#  - parse incoming JSON into Python objects
#  - validate types and field constraints
#  - generate OpenAPI documentation visible at /docs

class PredictRequest(BaseModel):
    """
    Payload sent by the client to /predict.

    Features are in the same order as the columns in the original CSV:
        index 0   : Time
        index 1-28: V1 through V28
        index 29  : Amount

    Values should be the raw transaction values — the server will apply
    the saved StandardScaler internally before running inference.
    """
    features: list[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 raw transaction features in CSV column order",
    )


class PredictResponse(BaseModel):
    """Response returned by /predict."""
    prediction:    str    # "FRAUD" or "LEGIT"
    probability:   float  # raw model output before thresholding
    threshold:     float  # decision threshold applied
    latency_ms:    float  # FHE inference time in milliseconds
    model:         str    # which model handled this request
    fhe_executed:  bool   # True if real FHE was used, False if plaintext


# ---------------------------------------------------------------------------
# Section 3 — Loading functions
# ---------------------------------------------------------------------------

def load_calibration_data() -> np.ndarray:
    """
    Load training features.  Used as calibration data for FHE compilation
    and to give Concrete-ML's `from_sklearn_model` enough samples to
    determine quantisation ranges.
    """
    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)

    # Tree models (decision_tree, xgboost) need the full value range to
    # correctly quantize split thresholds — subsampling causes the MLIR
    # compiler to mis-estimate bit widths and hit an assertion error.
    # NNs can work with 1000 samples because their ranges are set during
    # training; we subsample there only to keep compilation RAM manageable.
    if MODEL_NAME == "nn":
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=1000, replace=False)
        X_train = X_train[idx]
        print(f"  Calibration data loaded: {X_train.shape} "
              f"(subsampled for NN compilation)")
    else:
        print(f"  Calibration data loaded: {X_train.shape}")
    return X_train


def load_scaler():
    """
    Load the fitted StandardScaler that preprocessing.py saved.
    Used to scale incoming raw transactions the same way training data was scaled.
    """
    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_threshold() -> float:
    """Load the optimal decision threshold computed during model training."""
    with open(ARTIFACTS_DIR / "threshold.json") as f:
        return float(json.load(f)["threshold"])


def load_and_compile_model(X_calib: np.ndarray):
    """
    Load the saved CML model, then compile to an FHE circuit.

    All three model types are saved as cml_model.json using model.dump()
    BEFORE compile() is called during training.  This preserves float-valued
    split thresholds.  Calling compile() here re-quantizes from float to
    integers using X_calib, producing a correct FHE circuit.

    (The old approach of saving model.sklearn_model after compile() stored
    already-quantized integer thresholds; passing raw float features to that
    model always routed to the same leaf, making every prediction identical.)
    """
    print(f"\nLoading {MODEL_NAME} model from {ARTIFACTS_DIR}/")

    with open(ARTIFACTS_DIR / "cml_model.json") as f:
        cml_model = cml_load(f)

    if MODEL_NAME not in ("xgboost", "decision_tree", "nn"):
        raise RuntimeError(f"Unhandled model: {MODEL_NAME}")

    # Compile to FHE circuit — this is the slow step
    print(f"  Compiling to FHE circuit (this may take a minute)...")
    t0 = time.perf_counter()
    cml_model.compile(X_calib)
    elapsed = time.perf_counter() - t0
    print(f"  Compilation complete in {elapsed:.1f}s")

    # Generate the FHE evaluation key.  This is what makes encrypted
    # computation possible — without it, .predict(fhe="execute") fails.
    print(f"  Generating FHE keys...")
    t0 = time.perf_counter()
    cml_model.fhe_circuit.keygen()
    print(f"  Keys generated in {time.perf_counter() - t0:.1f}s")

    return cml_model


# ---------------------------------------------------------------------------
# Section 4 — Lifespan: runs once at startup and once at shutdown
# ---------------------------------------------------------------------------

# Globals populated at startup.  Marked None initially so type checkers
# know they may be unavailable before the lifespan event has fired.
cml_model = None
scaler    = None
threshold = None
metrics   = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — replaces the older @app.on_event("startup")
    pattern in modern FastAPI.

    Code before `yield` runs once at startup.
    Code after `yield` runs once at shutdown.
    """
    global cml_model, scaler, threshold, metrics

    print("=" * 60)
    print(f"Starting FHE Fraud Detection Server")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Compile model
    X_calib   = load_calibration_data()
    cml_model = load_and_compile_model(X_calib)

    # Load auxiliary artefacts
    scaler    = load_scaler()
    threshold = load_threshold()
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    print(f"\nServer ready.")
    print(f"  Decision threshold: {threshold:.4f}")
    print(f"  Reported PR-AUC   : {metrics.get('pr_auc', 'n/a')}")
    print("=" * 60)

    yield   # ← server handles requests during this period

    print("\nServer shutting down.")


# ---------------------------------------------------------------------------
# Section 5 — FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FHE Fraud Detection",
    description=(
        "Encrypted fraud detection inference using Fully Homomorphic "
        "Encryption via Concrete-ML."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Section 6 — Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Basic info endpoint."""
    return {
        "service": "FHE Fraud Detection",
        "model":   MODEL_NAME,
        "docs":    "/docs",
    }


@app.get("/health")
async def health():
    """
    Health check.  Returns 200 if the model is loaded and ready,
    503 otherwise.  Useful for load balancers or monitoring.
    """
    if cml_model is None:
        raise HTTPException(503, "Model not yet loaded")
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/info")
async def info():
    """Detailed model metadata and training metrics."""
    return {
        "model":     MODEL_NAME,
        "threshold": threshold,
        "metrics":   metrics,
        "n_features_expected": 30,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, fhe: bool = True, scaled: bool = False):
    if cml_model is None:
        raise HTTPException(503, "Server not ready")

    x = np.array(request.features, dtype=np.float32).reshape(1, -1)

    # Only scale if the client sent raw features
    if not scaled:
        x[:, [0, 29]] = scaler.transform(x[:, [0, 29]])

    fhe_mode = "execute" if fhe else "disable"

    t0      = time.perf_counter()
    y_proba = cml_model.predict_proba(x, fhe=fhe_mode)[0, 1]
    latency = (time.perf_counter() - t0) * 1000

    prediction = "FRAUD" if y_proba >= threshold else "LEGIT"

    return PredictResponse(
        prediction=prediction,
        probability=float(y_proba),
        threshold=threshold,
        latency_ms=latency,
        model=MODEL_NAME,
        fhe_executed=fhe,
    )


# ---------------------------------------------------------------------------
# Section 7 — Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # host=0.0.0.0 makes the server accessible from other machines on the
    # network.  Use 127.0.0.1 to bind to localhost only.
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # disable auto-reload — would re-compile on every save
    )