"""Flask dashboard for Customer Behavior Analytics.

    python app.py     ->  http://127.0.0.1:5000

On first run it creates demo data and trains the models automatically.
"""
import json
import threading
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for

import config
from src import features, scoring
from src.features import FEATURE_LABELS
from src.pipeline import artifacts_ready, run_training

app = Flask(__name__)
app.secret_key = "change-me-in-production"
_lock = threading.Lock()
STATE = {}


def load_state():
    if not artifacts_ready():
        print("No trained model found - training now (first run only)...")
        run_training()
    STATE["scores"] = pd.read_csv(config.SCORES_PATH, index_col="CustomerID")
    STATE["model"] = joblib.load(config.MODEL_PATH)
    STATE["metrics"] = json.loads(config.METRICS_PATH.read_text())


def read_history():
    if config.HISTORY_PATH.exists():
        try:
            return json.loads(config.HISTORY_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def log_lookup(cid, row):
    with _lock:
        hist = read_history()
        hist.insert(0, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "customer_id": int(cid),
                        "segment": row["segment"], "risk_level": row["risk_level"],
                        "risk_score": int(row["risk_score"])})
        config.HISTORY_PATH.write_text(json.dumps(hist[:200], indent=2))


def customer_payload(cid):
    scores = STATE["scores"]
    if cid not in scores.index:
        return None
    row = scores.loc[cid]
    return {
        "customer_id": int(cid), "segment": row["segment"],
        "churn_probability": float(row["churn_probability"]), "risk_score": int(row["risk_score"]),
        "risk_level": row["risk_level"], "priority": row["priority"],
        "reasons": row["reasons"].split(" | "), "suggested_action": row["suggested_action"],
        "features": {FEATURE_LABELS[f]: round(float(row[f]), 2) for f in features.FEATURES},
    }


@app.route("/")
def dashboard():
    s, m = STATE["scores"], STATE["metrics"]
    kpis = {
        "customers": len(s),
        "revenue": float(s["monetary"].sum()),
        "avg_churn": float(s["churn_probability"].mean()),
        "high_risk": int((s["risk_level"] == "High").sum()),
    }
    seg_counts = s["segment"].value_counts()
    risk_counts = s["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    seg_table = (s.groupby("segment").agg(customers=("recency", "size"), avg_recency=("recency", "mean"),
                                          avg_frequency=("frequency", "mean"), avg_spend=("monetary", "mean"),
                                          avg_churn=("churn_probability", "mean")).round(2).reset_index())
    # high-risk customers that are still worth saving: biggest spenders first
    top = (s[s["risk_level"] == "High"].sort_values("monetary", ascending=False).head(15)
           .reset_index()[["CustomerID", "segment", "monetary", "recency", "risk_score", "priority"]])
    return render_template("index.html", kpis=kpis, m=m, seg_table=seg_table.to_dict("records"),
                           top=top.to_dict("records"),
                           seg_labels=seg_counts.index.tolist(), seg_values=seg_counts.values.tolist(),
                           risk_labels=risk_counts.index.tolist(), risk_values=risk_counts.values.tolist())


@app.route("/lookup")
def lookup():
    raw = request.args.get("customer_id", "").strip()
    if not raw:
        return redirect(url_for("dashboard"))
    try:
        cid = int(raw)
    except ValueError:
        flash("Customer ID must be a number.")
        return redirect(url_for("dashboard"))
    return redirect(url_for("customer", cid=cid))


@app.route("/customer/<int:cid>")
def customer(cid):
    data = customer_payload(cid)
    if data is None:
        flash(f"Customer {cid} was not found.")
        return redirect(url_for("dashboard"))
    log_lookup(cid, STATE["scores"].loc[cid])
    return render_template("customer.html", c=data)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    defaults = {"recency": 30, "frequency": 6, "monetary": 600, "tenure_days": 300,
                "orders_last_90d": 2, "n_products": 15, "return_lines": 0}
    result, form = None, defaults
    if request.method == "POST":
        try:
            form = {k: float(request.form[k]) for k in defaults}
            if min(form.values()) < 0:
                raise ValueError("Values cannot be negative.")
            row = features.row_from_profile(**form)
            result = scoring.score_profile(STATE["model"], row, STATE["metrics"]["high_value_cutoff"])
        except (ValueError, KeyError) as exc:
            flash(f"Please enter valid numbers. {exc}")
    return render_template("predict.html", result=result, form=form)


@app.route("/history")
def history():
    return render_template("history.html", rows=read_history())


@app.route("/download/scores")
def download_scores():
    return send_file(config.SCORES_PATH, as_attachment=True, download_name="customer_scores.csv")


@app.route("/api/customer/<int:cid>")
def api_customer(cid):
    data = customer_payload(cid)
    if data is None:
        abort(404)
    return jsonify(data)


@app.route("/api/summary")
def api_summary():
    s = STATE["scores"]
    return jsonify({"customers": len(s), "segments": s["segment"].value_counts().to_dict(),
                    "risk_levels": s["risk_level"].value_counts().to_dict(),
                    "model": STATE["metrics"]["best_model"],
                    "metrics": STATE["metrics"]["model_results"][STATE["metrics"]["best_model"]]})


@app.template_filter("money")
def money(v):
    return f"{v:,.0f}"


load_state()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
