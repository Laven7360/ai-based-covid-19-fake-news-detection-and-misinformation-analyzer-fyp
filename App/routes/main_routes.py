from flask import render_template, Blueprint, redirect, url_for, request, flash
from flask import session, abort
from firebase_config import save_prediction
from flask import jsonify, current_app
import os, sys
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app_pipeline import classify_all_from_text, classify_article_from_url

# Initialize Blueprint
main_blueprint = Blueprint('main', __name__)

# === Path Setup (kept, though not used here) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))

# Home redirect -> single login page for all
@main_blueprint.route('/')
def home():
    return redirect(url_for('auth.login'))

@main_blueprint.route('/admin')
def admin_shortcut():
    if session.get("role") == "admin":
        return redirect(url_for('main.admin_dashboard'))
    return redirect(url_for('auth.login'))

# User dashboard
@main_blueprint.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

# Admin dashboard (guarded)
@main_blueprint.route('/admin_dashboard')
def admin_dashboard():
    if session.get("role") != "admin":
        abort(403)
    return render_template('admin_dashboard.html')

# ---------- News Analysis (Text) ----------
@main_blueprint.route('/news/text', methods=['GET', 'POST'])
def news_analysis_text():
    from app_pipeline import classify_all_from_text

    article_text = (request.args.get('q') or '').strip()
    result = None

    if request.method == 'POST':
       article_text = (request.form.get('article_text') or '').strip()
       if article_text:
          try:
              result = classify_all_from_text(article_text)
              try:
                  save_prediction(
                      user_id=session.get("user_id"),
                      text=article_text,
                      label=result.get("label"),
                      confidence_frac=(float(result.get("confidence", 0)) / 100.0),
                  )
              except Exception as e:
                  current_app.logger.warning("save_prediction(text) failed: %s", e)
          except Exception as e:
              result = {"label": "Unknown", "confidence": 0.0, "error": str(e)}

    return render_template('news_analysis_text.html',
                           article_text=article_text,
                           result=result)

# ---------- News Analysis (URL) ----------
@main_blueprint.route('/news/url', methods=['GET', 'POST'])
def news_analysis_url():
    article_url = (request.form.get('article_url') or '').strip() if request.method == 'POST' else ''
    if request.method == 'POST':
       result = classify_article_from_url(article_url, prefer="title")
       try:
           save_prediction(
               user_id=session.get("user_id"),
               text=result.get("title_used"),
               url=article_url,
               label=result.get("label"),
               confidence_frac=(float(result.get("confidence", 0)) / 100.0),
           )
       except Exception as e:
           current_app.logger.warning("save_prediction(url) failed: %s", e)

       return render_template('news_analysis_url.html', result=result, article_url=article_url)
    
    return render_template('news_analysis_url.html', article_url=article_url)

@main_blueprint.route('/verify', methods=['GET', 'POST'])
def fake_verify():
    from verifier import verify_claim
    result = None
    claim_text = (request.args.get('q') or '').strip()

    if request.method == 'POST':
       claim_text = (request.form.get('claim_text') or '').strip()
       if len(claim_text) < 8:
           result = {
               "verdict": "insufficient",
               "confidence": 0.0,
               "evidence": [],
               "error": "Please enter a longer claim (at least 8 characters)."
           }
       else:
           claim_text = claim_text[:1000]
           try:
               result = verify_claim(claim_text, k_pages=0, lang='en')
               try:
                   cls = classify_all_from_text(claim_text)  
                   save_prediction(
                       user_id=session.get("user_id"),
                       text=claim_text,
                       label=cls.get("label"),
                       confidence_frac=(float(cls.get("confidence", 0)) / 100.0),
                       verification={
                           "verdict": result.get("verdict"),
                           "confidence": float(result.get("confidence", 0.0)), 
                       },
                   )
               except Exception as e:
                   current_app.logger.warning("save_prediction(verify) failed: %s", e)
           except Exception as e:
               result = {
                   "verdict": "insufficient",
                   "confidence": 0.0,
                   "evidence": [],
                   "error": str(e)
               }

    return render_template('verify_claims.html', claim_text=claim_text, result=result)

# --- COVID-19 Trends (user) ---
import os, json

@main_blueprint.route("/trends", methods=["GET"])
def trends():
    return render_template("trends.html")

@main_blueprint.route("/trends/data", methods=["GET"])
def trends_data():
    """
    Serve Malaysia-only cumulative series from static/data/covid_cumulative.csv
    Expected columns (flexible): a date column (Date/Dates/Day) and a Malaysia cumulative column
    such as 'Malaysia', 'Malaysia_cumulative', 'Total_Malaysia', etc.
    """
    path = os.path.join(current_app.root_path, "static", "data", "covid_cumulative.csv")
    try:
        df = pd.read_csv(path)

        # Find a date column
        date_col = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("date", "dates", "day"):
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c
                    break
                except Exception:
                    pass
        if date_col is None:
            raise ValueError("No date column found in covid_cumulative.csv")

        # Find a Malaysia cumulative column
        m_cols = [c for c in df.columns
                  if "malay" in str(c).lower() and any(k in str(c).lower() for k in ("cum", "total", ""))]
        if not m_cols:
            raise ValueError("No Malaysia cumulative column found (e.g., 'Malaysia' / 'Malaysia_cumulative').")
        mcol = m_cols[0]

        # Clean and sort
        out = df[[date_col, mcol]].copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col)
        out[mcol] = pd.to_numeric(out[mcol], errors="coerce").fillna(method="ffill").fillna(0)

        # Build series
        dates = out[date_col].dt.strftime("%Y-%m-%d").tolist()
        malaysia = out[mcol].astype(float).tolist()
        malaysia_daily = out[mcol].astype(float).diff().fillna(0).clip(lower=0).tolist()

        return jsonify({"dates": dates, "malaysia": malaysia, "malaysia_daily": malaysia_daily})
    except Exception as e:
        # Small fallback so the page still renders if the CSV is missing
        dates = ["2020-01-01","2020-06-01","2021-01-01","2022-01-01","2023-01-01","2024-01-01"]
        malaysia = [0, 1000, 700000, 2800000, 5000000, 5000000]
        return jsonify({"dates": dates, "malaysia": malaysia, "malaysia_daily": [0]*len(dates), "error": str(e)})
    
@main_blueprint.app_errorhandler(403)
def forbidden(_):
    return render_template('403.html'), 403

