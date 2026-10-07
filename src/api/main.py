"""
FastAPI Inference Endpoint
===========================
REST API for crop disease prediction from RAW multispectral readings.

Clients send the 6 raw band reflectances per timestep; the server computes
the vegetation indices and applies the training scaler itself (via the
shared src.inference pipeline), so there is no train/serve skew.

Endpoints:
    POST /predict        one sequence  -> prediction
    POST /predict/batch  up to N sequences -> predictions
    GET  /health         200 if a model is loaded, else 503
    GET  /info           model architecture and training metadata
    GET  /metrics        Prometheus metrics

Configuration (environment variables):
    MODEL_CHECKPOINT            path to a checkpoint from src.training.train
    API_KEYS                    comma-separated keys accepted in X-API-Key
    API_ALLOW_UNAUTHENTICATED   "true" to disable auth (local development only)
    RATE_LIMIT_PER_MINUTE       per-client request budget (default 120)
    MAX_BODY_BYTES              request body cap (default 1_000_000)
    MAX_BATCH_SIZE              sequences per batch request (default 64)
    LOW_CONFIDENCE_THRESHOLD    below this, status = "low_confidence" (default 0.6)
    CORS_ORIGINS                comma-separated allowed origins (default: none)
"""

import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src.constants import BANDS, LABEL_NAMES
from src.inference.predictor import Predictor

logger = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────

def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ('1', 'true', 'yes')


class Settings:
    def __init__(self):
        self.checkpoint_path = os.environ.get('MODEL_CHECKPOINT',
                                              'models/checkpoints/lstm_best.pt')
        self.api_keys = {k.strip() for k in os.environ.get('API_KEYS', '').split(',')
                         if k.strip()}
        self.allow_unauthenticated = _env_bool('API_ALLOW_UNAUTHENTICATED')
        self.rate_limit_per_minute = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '120'))
        self.max_body_bytes = int(os.environ.get('MAX_BODY_BYTES', '1000000'))
        self.max_batch_size = int(os.environ.get('MAX_BATCH_SIZE', '64'))
        self.low_confidence = float(os.environ.get('LOW_CONFIDENCE_THRESHOLD', '0.6'))
        self.cors_origins = [o.strip() for o in os.environ.get('CORS_ORIGINS', '').split(',')
                             if o.strip()]


settings = Settings()


# ── Schemas ───────────────────────────────────────────────────

Reflectance = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Reading = Annotated[list[Reflectance], Field(min_length=len(BANDS), max_length=len(BANDS))]
Sequence = Annotated[list[Reading], Field(min_length=1, max_length=1000)]

EXAMPLE_READING = [0.35, 0.55, 0.65, 0.80, 0.20, 0.15]


class SpectralInput(BaseModel):
    """One sensor's time series of raw band reflectances."""
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={'examples': [{'readings': [EXAMPLE_READING] * 30}]},
    )
    readings: Sequence = Field(
        ...,
        description=f"Shape (seq_len, {len(BANDS)}). Each row is one timestep of raw "
                    f"reflectance in [0, 1], band order: {BANDS}. seq_len must equal "
                    "the model's training length (see GET /info).",
    )


class BatchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sequences: list[Sequence] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predicted_label: int
    label_name: str
    confidence: float
    risk_score: float = Field(description="Probability-weighted risk in [0, 1]")
    probabilities: list[float]
    status: Literal['ok', 'low_confidence', 'out_of_distribution']
    out_of_distribution: bool
    model_type: str
    inference_time_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded']
    model_loaded: bool
    device: str
    uptime_seconds: float
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_type: str
    input_features: list[str]
    raw_bands: list[str]
    seq_len: int
    hidden_dim: int
    num_layers: int
    num_classes: int
    labels: dict[int, str]
    total_parameters: int
    temperature: float
    validation_metrics: dict
    library_versions: dict
    device: str
    checkpoint_path: str


# ── State & metrics ───────────────────────────────────────────

class AppState:
    predictor: Predictor | None = None
    load_error: str | None = None
    start_time: float = time.time()


state = AppState()

registry = CollectorRegistry()
REQUESTS = Counter('api_requests_total', 'Requests by endpoint and status code',
                   ['endpoint', 'code'], registry=registry)
LATENCY = Histogram('api_inference_seconds', 'Model inference latency', registry=registry)
PREDICTIONS = Counter('api_predictions_total', 'Predictions by class and status',
                      ['label', 'status'], registry=registry)
MODEL_LOADED = Gauge('api_model_loaded', '1 if a model is loaded', registry=registry)


# ── Security: API key + rate limiting ─────────────────────────

api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


class RateLimiter:
    """Sliding-window limiter, per client identity, in process memory."""

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


rate_limiter = RateLimiter()


def authorize(request: Request, api_key: str | None = Depends(api_key_header)) -> str:
    """Require a valid X-API-Key (unless explicitly disabled) and apply the rate limit."""
    if settings.api_keys:
        if not api_key or not any(secrets.compare_digest(api_key, k) for k in settings.api_keys):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
        client = f"key:{api_key[:8]}"
    elif settings.allow_unauthenticated:
        client = f"ip:{request.client.host if request.client else 'unknown'}"
    else:
        raise HTTPException(status_code=503, detail="API authentication is not configured "
                            "(set API_KEYS, or API_ALLOW_UNAUTHENTICATED=true for local dev)")

    if not rate_limiter.check(client, settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes`, streamed or not."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        declared = dict(scope.get('headers') or []).get(b'content-length')
        if declared is not None and int(declared) > self.max_bytes:
            return await self._reject(send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send):
        await JSONResponse({'detail': 'Request body too large'}, status_code=413)(
            {'type': 'http'}, None, send)


class _BodyTooLarge(Exception):
    pass


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup. A failure leaves the API up but /health = 503."""
    state.start_time = time.time()
    state.predictor, state.load_error = None, None
    if not settings.api_keys and not settings.allow_unauthenticated:
        logger.warning("No API_KEYS configured: prediction endpoints will refuse requests")
    elif not settings.api_keys:
        logger.warning("API_ALLOW_UNAUTHENTICATED=true: prediction endpoints are OPEN")

    path = settings.checkpoint_path
    if not os.path.exists(path):
        state.load_error = f"Checkpoint not found: {path}"
    else:
        try:
            state.predictor = Predictor(path)
            logger.info("Loaded %s model from %s (seq_len=%d, T=%.3f) on %s",
                        state.predictor.model_type, path, state.predictor.seq_len,
                        state.predictor.temperature, state.predictor.device)
        except Exception as exc:
            logger.exception("Could not load model from %s", path)
            state.load_error = f"Model failed to load: {type(exc).__name__}"
    if state.load_error:
        logger.error(state.load_error)
    MODEL_LOADED.set(1 if state.predictor else 0)
    yield
    logger.info("Shutting down API")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Crop Disease Detection API",
    description="Predicts crop disease state from raw multispectral time series. "
                "Trained on synthetic data only -- see README limitations.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=['GET', 'POST'], allow_headers=['X-API-Key',
                                                                     'Content-Type'])


def require_model() -> Predictor:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.predictor


def _predict(predictor: Predictor, sequences: list[list[list[float]]]) -> list[PredictionResponse]:
    lengths = {len(s) for s in sequences}
    if lengths != {predictor.seq_len}:
        raise HTTPException(
            status_code=422,
            detail=f"Each sequence must have exactly {predictor.seq_len} timesteps "
                   f"(got lengths {sorted(lengths)})")

    start = time.perf_counter()
    try:
        with LATENCY.time():
            raw_results = predictor.predict_raw(np.asarray(sequences, dtype=np.float64))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc
    per_item_ms = (time.perf_counter() - start) * 1000 / len(sequences)

    out = []
    for r in raw_results:
        if r['out_of_distribution']:
            status = 'out_of_distribution'
        elif r['confidence'] < settings.low_confidence:
            status = 'low_confidence'
        else:
            status = 'ok'
        PREDICTIONS.labels(label=r['label_name'], status=status).inc()
        out.append(PredictionResponse(
            predicted_label=r['predicted_label'], label_name=r['label_name'],
            confidence=round(r['confidence'], 4), risk_score=round(r['risk_score'], 4),
            probabilities=[round(p, 4) for p in r['probabilities']], status=status,
            out_of_distribution=r['out_of_distribution'], model_type=predictor.model_type,
            inference_time_ms=round(per_item_ms, 2),
        ))
    return out


@app.middleware('http')
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    if request.url.path != '/metrics':
        REQUESTS.labels(endpoint=request.url.path, code=str(response.status_code)).inc()
    return response


@app.get('/health', response_model=HealthResponse)
def health_check(response: Response):
    """200 when the model is loaded and serving; 503 (degraded) otherwise."""
    loaded = state.predictor is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status='ok' if loaded else 'degraded',
        model_loaded=loaded,
        device=str(state.predictor.device) if loaded else 'n/a',
        uptime_seconds=round(time.time() - state.start_time, 1),
        detail=state.load_error,
    )


@app.get('/info', response_model=ModelInfoResponse)
def model_info(_: str = Depends(authorize), predictor: Predictor = Depends(require_model)):
    """Architecture, input contract and training metadata of the served model."""
    cfg = predictor.config
    return ModelInfoResponse(
        model_type=predictor.model_type,
        input_features=predictor.features.feature_names,
        raw_bands=BANDS,
        seq_len=predictor.seq_len,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_classes=len(LABEL_NAMES),
        labels=predictor.label_names,
        total_parameters=predictor.num_parameters,
        temperature=predictor.temperature,
        validation_metrics={k: v for k, v in predictor.metrics.items() if v is not None},
        library_versions=cfg.get('versions', {}),
        device=str(predictor.device),
        checkpoint_path=predictor.checkpoint_path,
    )


@app.post('/predict', response_model=PredictionResponse)
def predict(data: SpectralInput, _: str = Depends(authorize),
            predictor: Predictor = Depends(require_model)):
    """Predict disease state for one sequence of raw band readings."""
    return _predict(predictor, [data.readings])[0]


@app.post('/predict/batch', response_model=BatchResponse)
def predict_batch(data: BatchInput, _: str = Depends(authorize),
                  predictor: Predictor = Depends(require_model)):
    """Predict for several sequences at once (at most MAX_BATCH_SIZE)."""
    if len(data.sequences) > settings.max_batch_size:
        raise HTTPException(status_code=422,
                            detail=f"At most {settings.max_batch_size} sequences per batch")
    return BatchResponse(predictions=_predict(predictor, data.sequences))


@app.get('/metrics')
def metrics():
    """Prometheus metrics: request counts, inference latency, prediction mix."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
