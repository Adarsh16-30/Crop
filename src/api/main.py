"""
FastAPI Inference Endpoint
===========================
REST API for real-time crop disease prediction from spectral data.

Endpoints:
    POST /predict    — Run inference on a spectral sequence
    GET  /health     — Health check
    GET  /metrics    — Model performance metrics
    GET  /info       — System and model information
"""

import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.lstm_model import SpectralLSTM
from src.data.data_generator import LABEL_NAMES


# ── Request / Response Models ─────────────────────────────────

class SpectralInput(BaseModel):
    """Input schema for spectral sequence prediction."""
    sequence: list[list[float]] = Field(
        ...,
        description="Spectral time-series data. Shape: (timesteps, n_features). "
                    "Each row is one timestep with 12 features: "
                    "6 raw bands + 6 vegetation indices.",
        min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": [
                        [0.35, 0.55, 0.65, 0.80, 0.20, 0.15,
                         0.45, 0.60, 0.90, 0.78, 0.25, 1.03]
                    ] * 30
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for disease prediction."""
    predicted_label: int
    label_name: str
    confidence: float
    risk_score: float
    probabilities: list[float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_type: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    total_parameters: int
    device: str
    checkpoint_path: str | None


# ── Application State ─────────────────────────────────────────

class AppState:
    """Container for application state."""
    model: SpectralLSTM | None = None
    device: torch.device = torch.device('cpu')
    checkpoint_path: str | None = None
    start_time: float = 0.0
    request_count: int = 0


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    state.start_time = time.time()

    # Determine device
    if torch.cuda.is_available():
        state.device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        state.device = torch.device('cpu')
        print("Using CPU")

    # Load model
    checkpoint_path = os.environ.get(
        'MODEL_CHECKPOINT',
        'models/checkpoints/lstm_best.pt'
    )
    state.checkpoint_path = checkpoint_path

    if os.path.exists(checkpoint_path):
        try:
            state.model = SpectralLSTM(
                input_dim=12, hidden_dim=128,
                num_layers=2, num_classes=3
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=state.device,
                weights_only=False
            )
            state.model.load_state_dict(checkpoint['model_state_dict'])
            state.model.to(state.device)
            state.model.eval()
            print(f"Model loaded from: {checkpoint_path}")
            print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
            print(f"  Val loss: {checkpoint.get('val_loss', '?')}")
        except Exception as e:
            print(f"WARNING: Could not load model: {e}")
            state.model = None
    else:
        print(f"WARNING: Model checkpoint not found at {checkpoint_path}")
        print("API will return errors for /predict until a model is trained.")

    yield  # App runs here

    # Cleanup
    print("Shutting down API...")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="🌱 Crop Disease Detection API",
    description=(
        "REST API for quantum-enhanced crop disease detection. "
        "Accepts multispectral time-series data and returns disease predictions."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=str(state.device),
        uptime_seconds=round(time.time() - state.start_time, 1)
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model architecture information."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in state.model.parameters())
    return ModelInfoResponse(
        model_type="SpectralLSTM",
        input_dim=state.model.lstm.input_size,
        hidden_dim=state.model.hidden_dim,
        num_classes=3,
        total_parameters=total_params,
        device=str(state.device),
        checkpoint_path=state.checkpoint_path
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SpectralInput):
    """
    Run disease prediction on a spectral time-series sequence.

    The input sequence should have shape (timesteps, 12), where
    12 = 6 raw spectral bands + 6 vegetation indices:
    [red_edge_705, red_edge_740, red_edge_783, nir_842, swir_1610, swir_2190,
     NDRE, NDVI, MSI, SWIR_ratio, REIP_approx, CRI]
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with: "
                   "python -m src.training.train --model lstm"
        )

    start_time = time.time()

    try:
        # Convert to tensor
        x = torch.as_tensor(np.asarray(data.sequence, dtype=np.float32))
        x = x.unsqueeze(0).to(state.device, non_blocking=True)

        # Inference
        with torch.inference_mode():
            logits = state.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        label = int(np.argmax(probs))
        confidence = float(probs[label])

        # Risk score
        risk_map = {0: 0.0, 1: 0.5, 2: 1.0}
        risk_score = risk_map[label] * confidence

        inference_time = (time.time() - start_time) * 1000  # ms
        state.request_count += 1

        return PredictionResponse(
            predicted_label=label,
            label_name=LABEL_NAMES[label],
            confidence=round(confidence, 4),
            risk_score=round(risk_score, 4),
            probabilities=[round(float(p), 4) for p in probs],
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Get API usage metrics."""
    return {
        "total_requests": state.request_count,
        "uptime_seconds": round(time.time() - state.start_time, 1),
        "model_loaded": state.model is not None,
        "device": str(state.device)
    }
