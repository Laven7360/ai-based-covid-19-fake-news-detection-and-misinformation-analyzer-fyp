# App/verifier.py
import os, re, requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# ----------------------------------------------------------------------------- 
# .env + debug
# -----------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

FACTCHECK_API_KEY = (os.getenv("GOOGLE_FACTCHECK_API_KEY") or "").strip()
DEBUG = os.getenv("VERIFIER_DEBUG", "0") == "1"
def _log(*a):
    if DEBUG:
        print("[verifier]", *a)

SIMPLE_HEADERS = {"User-Agent": "TrueScopeAI/1.0 (mailto:you@example.com)"}

# ----------------------------------------------------------------------------- 
# Light NLP helpers
# -----------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
STOP = set("""
a an the of to in on for with by from this that those these and or as be is are was were been
being it its their his her they them we you your our not into over under at out up down off
""".split())
NEG_RE = re.compile(r"\b(no|not|never|none|cannot|can\s*not|does\s*not|do\s*not|did\s*not|is\s*not|are\s*not|won't|can't|n't)\b", re.I)

def split_sents(text: str, min_len: int = 30, max_len: int = 400) -> List[str]:
    if not text: return []
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for s in parts:
        s = s.strip()
        if not s: continue
        if len(s) > max_len:
            for c in re.split(r"\s*[;:]\s*|,\s+", s):
                c = c.strip()
                if len(c) >= min_len: out.append(c)
        elif len(s) >= min_len:
            out.append(s)
    return out[:12]

def _tokens(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return [t for t in s.split() if t and t not in STOP]

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _too_close_to_claim(sentence: str, claim: str, thr: float = 0.85) -> bool:
    return _jaccard(_tokens(sentence), _tokens(claim)) >= thr

def _normalize_neg(s: str) -> str:
    s = re.sub(r"\bdoesn[’']?t\b", "does not", s, flags=re.I)
    s = re.sub(r"\bdon[’']?t\b", "do not", s, flags=re.I)
    s = re.sub(r"\bdidn[’']?t\b", "did not", s, flags=re.I)
    s = re.sub(r"\bisn[’']?t\b", "is not", s, flags=re.I)
    s = re.sub(r"\baren[’']?t\b", "are not", s, flags=re.I)
    s = re.sub(r"\bcan[’']?t\b", "cannot", s, flags=re.I)
    return s

def _remove_neg_words(s: str) -> str:
    # drop explicit negators to form a "positive" variant
    return NEG_RE.sub(" ", s)

# ----------------------------------------------------------------------------- 
# Sentence embeddings (semantic similarity)
# -----------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None   # type: ignore[assignment]
    np = None

_enc = None  # lazy-loaded global

def _get_encoder():
    """Return a global MiniLM encoder instance (lazy init)."""
    global _enc
    if _enc is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        _enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _enc

def _cos_sim(a, b) -> float:
    if np is None: return 0.0
    if a is None or b is None: return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0: return 0.0
    return float(np.dot(a, b) / denom)

def _embed(texts: List[str]):
    enc = _get_encoder()
    if enc is None: return None
    return enc.encode(texts, normalize_embeddings=True)

# ----------------------------------------------------------------------------- 
# NLI (entail/contradict) with heuristic fallback
# -----------------------------------------------------------------------------
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
_HAS_NLI = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    _HAS_NLI = False
    torch = None

NLI_MODEL = "cross-encoder/nli-distilroberta-base"
_device = _tokenizer = _nli = _id2label = None

def _init_nli() -> bool:
    global _device, _tokenizer, _nli, _id2label
    if not _HAS_NLI: return False
    if _nli is not None: return True
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(_device).eval()
        _id2label = getattr(_nli.config, "id2label", None)
        return True
    except Exception as e:
        _log("NLI load failed:", e)
        return False

def _nli_probs(premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
    if not _init_nli(): return None
    a, b = (premise or "").strip(), (hypothesis or "").strip()
    if not a or not b: return None
    with torch.no_grad():
        enc = _tokenizer(a, b, return_tensors="pt", truncation=True, max_length=384).to(_device)
        logits = _nli(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    idx = {"entailment": None, "contradiction": None, "neutral": None}
    if _id2label:
        for i, name in _id2label.items():
            name = str(name).lower()
            for key in list(idx.keys()):
                if key in name:
                    idx[key] = int(i)
    if None in idx.values():
        idx = {"contradiction": 0, "neutral": 1, "entailment": 2}
    return {
        "entailment": float(probs[idx["entailment"]]),
        "contradiction": float(probs[idx["contradiction"]]),
        "neutral": float(probs[idx["neutral"]]),
    }

def _heuristic_score(prem: str, hyp: str) -> Dict[str, float]:
    tp = set(_tokens(prem)); th = set(_tokens(hyp))
    overlap = len(tp & th) / max(1, len(th))
    neg_p = bool(NEG_RE.search(prem))
    neg_h = bool(NEG_RE.search(hyp))
    if overlap >= 0.5 and (neg_p != neg_h):
        return {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1}
    if overlap >= 0.6 and (neg_p == neg_h):
        return {"entailment": 0.7, "contradiction": 0.1, "neutral": 0.2}
    return {"entailment": 0.3 * overlap, "contradiction": 0.2 * overlap, "neutral": 1 - 0.5 * overlap}

def _score_pair(prem: str, hyp: str) -> Dict[str, float]:
    return _nli_probs(prem, hyp) or _heuristic_score(prem, hyp)

# ----------------------------------------------------------------------------- 
# Rating helpers (publisher verdicts)
# -----------------------------------------------------------------------------
FALSE_TOKENS = ("false", "mostly false", "pants on fire", "misleading", "incorrect", "partly false", "untrue", "no evidence", "flawed")
TRUE_TOKENS  = ("true", "mostly true", "accurate", "correct")

def _rating_group(textual_rating: str) -> str:
    t = (textual_rating or "").strip().lower()
    if any(k in t for k in TRUE_TOKENS):  return "true"
    if any(k in t for k in FALSE_TOKENS): return "false"
    if t: return "mixed"
    return "unknown"

# ----------------------------------------------------------------------------- 
# Fetch page text (for snippet only)
# -----------------------------------------------------------------------------
from bs4 import BeautifulSoup
def _fetch_text(url: str) -> str:
    try:
        r = requests.get(url, headers=SIMPLE_HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        txt = " ".join(p for p in paras if p and len(p) > 30)
        return txt[:6000]
    except Exception:
        return ""

def _best_sentence(sentences: List[str], claim: str) -> Optional[str]:
    claim_norm = _normalize_neg(claim)
    best = None
    best_score = -1.0
    for s in sentences:
        if _too_close_to_claim(s, claim_norm):
            continue
        score = min(len(s), 220) / 220.0 + 0.15 * _jaccard(_tokens(s), _tokens(claim_norm))
        if score > best_score:
            best, best_score = s, score
    return best

# ----------------------------------------------------------------------------- 
# Query strategy for the Google Fact Check API
# -----------------------------------------------------------------------------
def _keyword_query(s: str, limit: int = 12) -> str:
    toks = [t for t in _tokens(s) if len(t) > 2]
    seen, out = set(), []
    for t in toks:
        if t in seen: continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit: break
    return " ".join(out) if out else s

def _generate_queries(claim: str) -> List[str]:
    claim = claim.strip()
    norm = _normalize_neg(claim)
    positive = _remove_neg_words(norm)  
    kw = _keyword_query(norm, limit=12)
    queries = [claim, norm, positive, kw]
    seen, out = set(), []
    for q in queries:
        q = (q or "").strip()
        if not q: continue
        if q in seen: continue
        seen.add(q); out.append(q)
    return out

# ----------------------------------------------------------------------------- 
# Google Fact Check Tools API (multi-query + similarity)
# -----------------------------------------------------------------------------
def _from_google_factcheck(claim: str, lang: str = "en", max_items: int = 10) -> List[Dict[str, Any]]:
    key = FACTCHECK_API_KEY
    if not key:
        _log("No GOOGLE_FACTCHECK_API_KEY set")
        return []

    queries = _generate_queries(claim)
    _log("queries:", queries)

    # Pre-embed user claim once for similarity (best effort)
    claim_vec = None
    try:
        claim_vec = _embed([claim])[0]  
    except Exception as e:
        _log("embed disabled:", e)
        claim_vec = None

    candidates: List[Dict[str, Any]] = []
    seen_urls = set()

    for q in queries:
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {"query": q, "languageCode": lang, "key": key}
            r = requests.get(url, params=params, headers=SIMPLE_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json() or {}
        except Exception as e:
            _log("FactCheck API error for query:", q, e)
            continue

        for c in data.get("claims", []):
            reviewed_text = (c.get("text") or "").strip()
            reviews = c.get("claimReview") or []
            for rev in reviews:
                u = (rev.get("url") or "").strip()
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)

                title = (rev.get("title") or "").strip()
                publisher = ((rev.get("publisher") or {}).get("name") or urlparse(u).netloc.replace("www.",""))
                rating = (rev.get("textualRating") or "").strip()

                # semantic similarity (user <-> reviewed text/title)
                sim = 0.0
                if claim_vec is not None:
                    pieces = [p for p in [reviewed_text, title] if p]
                    if pieces:
                        vecs = _embed(pieces)
                        sim = max(_cos_sim(claim_vec, vecs[i]) for i in range(len(pieces)))

                if (claim_vec is not None) and reviewed_text and (sim < 0.45):
                    continue

                candidates.append({
                    "url": u,
                    "publisher": publisher,
                    "title": title or publisher,
                    "reviewed_text": reviewed_text,
                    "rating": rating or None,
                    "similarity": round(sim, 3),
                })
                if len(candidates) >= max_items:
                    break
            if len(candidates) >= max_items:
                break
        if len(candidates) >= max_items:
            break

    _log("candidates:", len(candidates))
    candidates.sort(key=lambda d: d.get("similarity", 0.0), reverse=True)
    return candidates

# ----------------------------------------------------------------------------- 
# Model fallback (only used when the FactCheck API finds no evidence)
# -----------------------------------------------------------------------------
def _model_suggestion(claim: str) -> Optional[Dict[str, Any]]:
    """
    Calls your existing classifier to get a 'fake/real' suggestion.
    Returns a dict like: {'label': 'fake'|'real'|'unknown', 'prob': <float or None>}
    If anything fails, returns None (so UI won’t show the block).
    """
    try:
        # Import lazily to avoid circular imports at module import time
        from app_pipeline import classify_all_from_text
    except Exception as e:
        _log("model hook import failed:", e)
        return None

    try:
        res = classify_all_from_text(claim)
    except Exception as e:
        _log("model prediction failed:", e)
        return None

    label_str = None
    prob_val = None

    if isinstance(res, dict):
        # common label keys
        for k in ("final_label", "label", "prediction", "verdict", "result", "class"):
            v = res.get(k)
            if isinstance(v, str) and v.strip():
                label_str = v.strip().lower()
                break

        # common probability keys
        for k in ("prob_fake", "fake_prob", "p_fake", "pred_prob", "prob"):
            v = res.get(k)
            if isinstance(v, (int, float)):
                prob_val = float(v)
                break

    if not label_str:
        label_str = str(res).strip().lower()

    # Normalize to 'fake' or 'real' if possible
    out_label = None
    if "fake" in label_str and "real" not in label_str:
        out_label = "fake"
    elif ("real" in label_str) or ("true" in label_str):
        out_label = "real"

    if out_label is None and isinstance(prob_val, (int, float)):
        out_label = "fake" if prob_val >= 0.5 else "real"

    if out_label is None:
        out_label = "unknown"

    return {"label": out_label, "prob": prob_val}

# ----------------------------------------------------------------------------- 
# Public API
# -----------------------------------------------------------------------------
def verify_claim(claim: str, k_pages: int = 0, lang: str = "en") -> Dict[str, Any]:
    """
    Multi-query FactCheck API + semantic similarity + NLI alignment.
    Handles original text, paraphrases, and negations more robustly.
    """
    claim = _normalize_neg((claim or "").strip())
    if not claim:
        return {"verdict": "insufficient", "confidence": 0.0, "evidence": [], "error": "Empty claim."}

    rows = _from_google_factcheck(claim, lang=lang, max_items=12)
    if not rows:
        pred = _model_suggestion(claim)  
        return {
        "verdict": "insufficient",
        "confidence": 0.0,
        "evidence": [],
        "model_pred": pred,  
    }

    evidence: List[Dict[str, Any]] = []
    best_support = best_refute = 0.0

    for row in rows:
        reviewed = row.get("reviewed_text") or row.get("title") or ""
        probs = _score_pair(reviewed, claim)
        ent, con = float(probs["entailment"]), float(probs["contradiction"])
        grp = _rating_group(row.get("rating") or "")

        # rating-aware mapping
        if grp == "false":
            support = con
            refute  = ent
        elif grp == "true":
            support = ent
            refute  = con
        else:
            support = max(ent, 0.0)
            refute  = max(con, 0.0)

        # snippet
        page_text = _fetch_text(row["url"])
        snippet = _best_sentence(split_sents(page_text), claim) or (reviewed[:300] if reviewed else row["title"][:300])

        ev = {
            "source": f"{row['publisher']} (FactCheck API)",
            "title": row["title"],
            "url": row["url"],
            "rating": row.get("rating"),
            "snippet": snippet,
            "entailment": round(ent, 3),
            "contradiction": round(con, 3),
            "similarity": row.get("similarity", 0.0),
            "support_score": round(support, 3),
            "refute_score": round(refute, 3),
        }
        evidence.append(ev)
        best_support = max(best_support, support)
        best_refute  = max(best_refute,  refute)

    HI, MARGIN = 0.60, 0.15
    margin = abs(best_support - best_refute)
    if (best_support >= HI) and ((best_support - best_refute) >= MARGIN):
        verdict = "supports"
        conf = min(1.0, best_support * (0.75 + 0.25 * margin))
    elif (best_refute >= HI) and ((best_refute - best_support) >= MARGIN):
        verdict = "refutes"
        conf = min(1.0, best_refute * (0.75 + 0.25 * margin))
    else:
        verdict = "insufficient"
        conf = 0.35 + 0.25 * max(best_support, best_refute)

    evidence.sort(key=lambda e: max(e["support_score"], e["refute_score"], e.get("similarity", 0.0)), reverse=True)
    support_best = next((e for e in evidence if e["support_score"] >= e["refute_score"]), None)
    refute_best  = next((e for e in evidence if e["refute_score"]  >  e["support_score"]), None)

    selected: List[Dict[str, Any]] = []
    if support_best: selected.append(support_best)
    if refute_best and (not selected or selected[0]["url"] != refute_best["url"]):
        selected.append(refute_best)
    for e in evidence:
        if len(selected) >= 2: break
        if all(e["url"] != s["url"] for s in selected):
            selected.append(e)

    return {"verdict": verdict, "confidence": round(conf, 3), "evidence": selected}
