"""
main.py — FastAPI entrypoint.

Endpoints:
  POST /predict   -> run the real-time phishing model on a URL, log result
  GET  /history   -> paginated/searchable prediction history
  GET  /dashboard -> aggregate stats for the analytics dashboard
"""

import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import init_db, get_db, PredictionLog
from predictor import predict_url

app = FastAPI(title="Real-Time Phishing Detection API")

# Allow the Vercel-hosted frontend (and local dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend domain before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)


class PredictRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if len(v) > 2048:
            raise ValueError("URL is too long")
        candidate = v if re.match(r"^https?://", v, re.IGNORECASE) else "http://" + v
        if not _URL_PATTERN.match(candidate):
            raise ValueError("Invalid URL format")
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    reasons: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    try:
        result = predict_url(payload.url)
    except Exception as e:
        # Don't leak internals; log e server-side if you add logging
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    log = PredictionLog(
        url=payload.url,
        prediction=result["prediction"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        latency=result["latency_ms"],
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()

    return result


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@app.get("/history")
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    prediction_filter: Optional[str] = Query(None, pattern="^(phishing|legitimate)$"),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionLog)

    if search:
        query = query.filter(PredictionLog.url.ilike(f"%{search}%"))
    if prediction_filter:
        query = query.filter(PredictionLog.prediction == prediction_filter)

    total = query.count()
    items = (
        query.order_by(PredictionLog.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": i.id,
                "url": i.url,
                "prediction": i.prediction,
                "confidence": i.confidence,
                "risk_level": i.risk_level,
                "latency_ms": i.latency,
                "timestamp": i.timestamp.isoformat(),
            }
            for i in items
        ],
    }


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------
@app.get("/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    total_scans = db.query(func.count(PredictionLog.id)).scalar() or 0
    phishing_count = (
        db.query(func.count(PredictionLog.id))
        .filter(PredictionLog.prediction == "phishing")
        .scalar()
        or 0
    )
    legitimate_count = total_scans - phishing_count
    avg_latency = db.query(func.avg(PredictionLog.latency)).scalar() or 0.0

    return {
        "total_scans": total_scans,
        "phishing_count": phishing_count,
        "legitimate_count": legitimate_count,
        "average_latency_ms": round(float(avg_latency), 1),
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "Real-Time Phishing Detection API"}