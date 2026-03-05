# routes/admin_routes.py
from flask import Blueprint, jsonify, session, abort, request
from datetime import datetime, timedelta
from firebase_admin import firestore
from firebase_config import db

from google.cloud.firestore_v1 import Query

admin_blueprint = Blueprint("admin", __name__)


# -------- Guard: admin only
def admin_required():
    if session.get("role") != "admin":
        abort(403)

def _count(query: Query) -> int:
    try:
        agg = query.count()
        res = list(agg.get())
        # Works across firestore versions
        try:
            return int(res[0][0].value) if res else 0
        except Exception:
            # Some older versions expose a single .value
            return int(getattr(res[0], "value", 0)) if res else 0
    except Exception:
        # Minimal-memory fallback
        return sum(1 for _ in query.stream())

def _today_utc():
    """Returns the current UTC date, with time set to 00:00:00"""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

# -------- KPIs + trend (last 30 days)
@admin_blueprint.route("/admin/stats", methods=["GET"])
def admin_stats():
    admin_required()

    users_total = _count(db.collection("users").where("role", "==", "user"))
    print(f"Total Users: {users_total}") 

    subs_q = db.collection("predictions")
    submissions_total = _count(subs_q)
    fake_total = _count(subs_q.where("label", "==", "Fake"))
    real_total = _count(subs_q.where("label", "==", "Real"))

    agree_q1 = subs_q.where("label", "==", "Real").where("verification.verdict", "==", "supports")
    agree_q2 = subs_q.where("label", "==", "Fake").where("verification.verdict", "==", "refutes")
    agree = _count(agree_q1) + _count(agree_q2)
    with_verif = _count(subs_q.where("verification.verdict", "in", ["supports", "refutes", "insufficient"]))

    agreement_rate = (agree / with_verif) if with_verif else 0.0

    # Build 30-day trend
    start = _today_utc() - timedelta(days=29)
    try:
        trend_docs = subs_q.where("created_at", ">=", start).order_by("created_at").get()
    except Exception:
        trend_docs = subs_q.where("created_at", ">=", start).get()

    days = [(start + timedelta(days=i)).date() for i in range(30)]
    idx = {d: {"real":0, "fake":0, "supports":0, "refutes":0, "insufficient":0} for d in days}
    for doc in trend_docs:
        d = doc.to_dict()
        dt = d.get("created_at")
        if isinstance(dt, datetime):
            day = dt.date()
            if day in idx:
                if d.get("label") == "Real": idx[day]["real"] += 1
                if d.get("label") == "Fake": idx[day]["fake"] += 1
                ver = (d.get("verification") or {}).get("verdict")
                if ver in idx[day]: idx[day][ver] += 1

    trend = {
        "dates": [d.isoformat() for d in days],
        "real": [idx[d]["real"] for d in days],
        "fake": [idx[d]["fake"] for d in days],
        "supports": [idx[d]["supports"] for d in days],
        "refutes": [idx[d]["refutes"] for d in days],
        "insufficient": [idx[d]["insufficient"] for d in days],
    }

    return jsonify({
        "totals": {
            "users": users_total,
            "submissions": submissions_total,
            "fake": fake_total,
            "real": real_total,
            "agreement_rate": round(agreement_rate, 4)
        },
        "trend_30d": trend
    })

# -------- Recent activity table
@admin_blueprint.route("/admin/activity", methods=["GET"])
def admin_activity():
    admin_required()
    limit = int(request.args.get("limit", 20))
    q = db.collection("predictions").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    rows = []
    for doc in q.get():
        d = doc.to_dict()
        ver = (d.get("verification") or {})
        created = d.get("created_at")
        rows.append({
            "created_at": created.isoformat() if isinstance(created, datetime) else "",
            "user_id": d.get("user_id") or "-",
            "snippet": (d.get("snippet") or d.get("text_preview") or "")[:140],
            "label": d.get("label") or "-",
            "verification": ver.get("verdict") or "-",
            "confidence": d.get("confidence"),
        })
    return jsonify({"rows": rows})

# -------- COVID chart data 
@admin_blueprint.route("/admin/covid", methods=["GET"])
def admin_covid():
    admin_required()
    return jsonify({
        "updated": None,
        "series": {
            "dates": [],
            "new_cases_smoothed": [],
            "new_deaths_smoothed": []
        }
    })
