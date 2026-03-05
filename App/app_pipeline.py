import os
import json
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import torch

try:
    from flask import current_app
except Exception:  
    current_app = None

# ---------- newspaper3k (for URL fetching; optional) ----------
try:
    from newspaper import Article, Config
    _NP_CONFIG = Config()
    _NP_CONFIG.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/124.0"
    )
    _NP_CONFIG.request_timeout = 10
except Exception:  
    Article = None
    _NP_CONFIG = None

# ---------- Paths / project layout ----------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# ---------- small helper ----------
def _cfg(key: str, default=None):
    """Read from Flask app.config if present, else environment, else default."""
    if current_app and getattr(current_app, "config", None) and (key in current_app.config):
        return current_app.config.get(key, default)
    return os.getenv(key, default)

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or ""))

# ---------- RoBERTa configuration ----------
ROBERTA_MODEL_DIR = _cfg(
    "MODEL_DIR",
    str(PROJECT / "Model" / "BERT" / "outputs_robertabert" / "best_model"),
)
ROBERTA_METRICS = _cfg(
    "METRICS_JSON",
    str(PROJECT / "Model" / "BERT" / "outputs_robertabert" / "robertabert_metrics.json"),
)
ROBERTA_MAX_LEN = int(_cfg("MAX_LEN", "320"))
ROBERTA_DEVICE  = _cfg("INFER_DEVICE", "cpu")

# Optional runtime overrides
THRESHOLD_OVERRIDE = _cfg("THRESHOLD", None)     
TEMP_OVERRIDE      = _cfg("TEMPERATURE", None)   

# ---------- Lazy singletons ----------
_tokenizer   = None
_roberta     = None
_threshold   = None
_temperature = None
_device      = None

def _load_roberta() -> bool:
    """
    Initialize tokenizer/model once, and read calibration params:
      - best_threshold
      - temperature
    from robertabert_metrics.json (or env overrides).
    """
    global _tokenizer, _roberta, _threshold, _temperature, _device
    if _roberta is not None:
        return True

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        return False

    model_dir = Path(ROBERTA_MODEL_DIR)
    if not model_dir.exists():
        return False

    try:
        # device
        _device = torch.device(
            ROBERTA_DEVICE if ROBERTA_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # model + tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _roberta   = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(_device).eval()

        # defaults
        thr  = 0.5
        temp = 1.0

        # read metrics.json saved by training pipeline
        mpath = Path(ROBERTA_METRICS) if ROBERTA_METRICS else None
        if mpath and mpath.exists():
            try:
                with open(mpath, "r") as f:
                    meta = json.load(f)
                thr  = float(meta.get("best_threshold", thr))
                temp = float(meta.get("temperature",   temp))
            except Exception:
                pass

        # env/config overrides
        if THRESHOLD_OVERRIDE is not None:
            thr = float(THRESHOLD_OVERRIDE)
        if TEMP_OVERRIDE is not None:
            temp = float(TEMP_OVERRIDE)

        _threshold   = float(thr)
        _temperature = max(1e-6, float(temp))
        return True

    except Exception:
        _tokenizer = _roberta = _device = _threshold = _temperature = None
        return False

@torch.no_grad()
def _predict_text_roberta(text: str):
    """
    Predict using RoBERTa with temperature scaling.
    Returns (label: 'Real'|'Fake', confidence: float in [0,1]).
    """
    if not _load_roberta():
        raise RuntimeError("RoBERTa not available")

    enc = _tokenizer(
        _nfkc(text),
        return_tensors="pt",
        truncation=True,
        max_length=ROBERTA_MAX_LEN,
    ).to(_device)

    logits = _roberta(**enc).logits
    # temperature scaling learned on validation during training
    probs  = torch.softmax(logits / _temperature, dim=-1)
    p_real = probs[0, 1].item()

    label = "Real" if p_real >= _threshold else "Fake"
    conf  = p_real if label == "Real" else 1.0 - p_real
    return label, float(conf)

# ---------- Public wrapper used by your routes ----------
def _predict_text(text: str):
    try:
        return _predict_text_roberta(text)
    except Exception:
        return "Unknown", 0.0

# ---------- URL fetch helpers (title/text) ----------
def _fetch_article_title(url: str):
    if Article is None:
        return None
    try:
        art = Article(url, config=_NP_CONFIG) if _NP_CONFIG else Article(url)
        art.download(); art.parse()
        title = (art.title or "").strip()
        return title or None
    except Exception:
        return None

def _fetch_article_text(url: str):
    if Article is None:
        return None
    try:
        art = Article(url, config=_NP_CONFIG) if _NP_CONFIG else Article(url)
        art.download(); art.parse()
        text = (art.text or "").strip()
        return text or None
    except Exception:
        return None

# ---------- Public API ----------
def classify_all_from_text(text: str):
    label, p = _predict_text(text or "")
    return {"label": label, "confidence": round(max(0.0, min(1.0, p)) * 100, 2)}

def classify_article_from_url(url: str, prefer: str = "title"):
    """
    prefer='title' -> try title first (faster/cleaner); fall back to text if missing.
    prefer='text'  -> fetch full text first; fall back to title if missing.
    """
    domain = urlparse(url).netloc.replace("www.", "")
    title = text = None

    if prefer == "text":
        text  = _fetch_article_text(url)
        title = _fetch_article_title(url) or None
    else:
        title = _fetch_article_title(url)
        if not title:
            text = _fetch_article_text(url)

    candidate = title or text
    if not candidate:
        return {
            "error": "Failed to fetch article content.",
            "label": "Unknown", "confidence": 0.0, "source_domain": domain
        }

    label, p = _predict_text(candidate)
    out = {
        "label": label,
        "confidence": round(max(0.0, min(1.0, p)) * 100, 2),
        "source_domain": domain
    }
    if title:
        out["title_used"] = title
    return out
