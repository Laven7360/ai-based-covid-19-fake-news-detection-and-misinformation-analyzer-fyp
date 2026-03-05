# App/firebase_config.py
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Optional, Dict, Any

# Initialize Admin SDK once, reuse everywhere
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------- Small helpers call from routes ----------

def _norm_label(label: Optional[str]) -> str:
    s = (label or "").strip().lower()
    if s.startswith("real") or "true" in s:
        return "Real"
    if s.startswith("fake") or "false" in s:
        return "Fake"
    return "Unknown"

def _clip01(x: Optional[float]) -> float:
    try:
        return max(0.0, min(1.0, float(x or 0.0)))
    except Exception:
        return 0.0

def save_prediction(
    *,
    user_id: Optional[str],
    text: Optional[str] = None,
    url: Optional[str] = None,
    label: Optional[str],
    confidence_frac: Optional[float],    
    verification: Optional[Dict[str, Any]] = None
) -> str:
    """Create a prediction doc. Returns doc id."""
    doc = {
        "created_at": datetime.utcnow(),
        "user_id": user_id or None,
        "source_type": "url" if url else "text",
        "url": url or None,
        "text_preview": (text or "")[:280] or None,
        "snippet": (text or url or "")[:280],
        "label": _norm_label(label),
        "confidence": _clip01(confidence_frac),
    }
    if verification:
        v = {
            "verdict": (verification.get("verdict") or "").strip(),
            "confidence": _clip01(verification.get("confidence")),
        }
        doc["verification"] = v

    ref = db.collection("predictions").add(doc)[1]
    return ref.id
