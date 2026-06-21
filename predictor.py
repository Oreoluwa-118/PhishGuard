"""
predictor.py
 
Loads realtime_model.pkl + realtime_scaler.pkl and exposes predict_url(url),
matching the response shape the rest of the app (API/frontend/DB) already expects:
 
{
    "prediction": "phishing" | "legitimate",
    "confidence": float (0-100),
    "risk_level": "low" | "medium" | "high",
    "reasons": [str, ...],
    "latency_ms": float
}
"""
 
import os
import time
import joblib
import pandas as pd
 
from feature_extractor import extract_features, FEATURE_ORDER
 
# ---------------------------------------------------------------------------
# Load model + scaler once at startup
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
 
model = joblib.load(os.path.join(_MODELS_DIR, "realtime_model.pkl"))
scaler = joblib.load(os.path.join(_MODELS_DIR, "realtime_scaler.pkl"))
 
# classes_ confirmed as [0, 1] when we inspected the model.
# CONFIRM which integer means which label against your training notebook —
# this mapping is assumed, not verified.
LABELS = {0: "legitimate", 1: "phishing"}
 
 
def _risk_level(confidence: float, prediction: str) -> str:
    if prediction == "legitimate":
        # high confidence legitimate = low risk; low confidence legitimate = medium risk
        return "low" if confidence >= 70 else "medium"
    # phishing
    if confidence >= 85:
        return "high"
    if confidence >= 60:
        return "medium"
    return "low"
 
 
def _build_reasons(features: dict, prediction: str) -> list:
    reasons = []
 
    if prediction == "phishing":
        if features.get("custom_uses_http"):
            reasons.append("No HTTPS detected")
        if features.get("custom_has_ip"):
            reasons.append("IP address used instead of a domain name")
        if features.get("custom_has_suspicious_kw"):
            reasons.append("Suspicious keyword detected in URL")
        if features.get("brand_path_not_domain"):
            reasons.append("Brand name appears in path/subdomain but not in the actual domain")
        if features.get("custom_has_typosquat") or features.get("brand_is_typosquat"):
            reasons.append("Domain closely resembles a known brand (possible typosquatting)")
        if features.get("tld_is_high_risk"):
            reasons.append("Domain uses a TLD commonly associated with abuse")
        if features.get("custom_is_free_hosting"):
            reasons.append("Hosted on a free hosting platform")
        if features.get("custom_domain_has_hyphen") and features.get("brand_in_domain"):
            reasons.append("Hyphenated domain combined with brand name")
        if features.get("lex_has_random_string"):
            reasons.append("Domain contains a randomly generated-looking string")
        if features.get("combo_login_no_https"):
            reasons.append("Login page without HTTPS")
        if not reasons:
            reasons.append("Model flagged this URL based on its overall feature pattern")
    else:
        if not features.get("custom_uses_http"):
            reasons.append("HTTPS detected")
        if not features.get("custom_has_suspicious_kw"):
            reasons.append("No suspicious keywords found")
        if features.get("tld_is_common_legit"):
            reasons.append("Uses a commonly trusted top-level domain")
        if not reasons:
            reasons.append("No major risk indicators found")
 
    return reasons
 
 
def predict_url(url: str) -> dict:
    start = time.time()
 
    features = extract_features(url)
    X = pd.DataFrame([features], columns=FEATURE_ORDER)
    X_scaled = scaler.transform(X)
 
    pred_class = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)
 
    prediction = LABELS[pred_class]
    risk_level = _risk_level(confidence, prediction)
    reasons = _build_reasons(features, prediction)
 
    latency_ms = round((time.time() - start) * 1000, 1)
 
    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "reasons": reasons,
        "latency_ms": latency_ms,
    }
 