"""End-to-end training pipeline: data -> features -> models -> scores."""
import json

import joblib
import pandas as pd

import config
from src import churn_model, features, plots, preprocess, scoring, segmentation


def run_training(k=None, verbose=True):
    log = print if verbose else (lambda *a, **kw: None)
    for d in (config.MODEL_DIR, config.OUTPUT_DIR, config.PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    path, source = preprocess.resolve_data_path()
    log(f"[1/6] Loading {source} data: {path.name}")
    raw = preprocess.load_raw(path)
    sales, returns = preprocess.clean(raw)
    log(f"      {len(raw):,} raw rows -> {len(sales):,} valid sales lines, "
        f"{sales['CustomerID'].nunique():,} customers")

    horizon = pd.Timedelta(days=config.CHURN_WINDOW_DAYS)
    end = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)   # exclusive end
    snap_test, snap_train = end - horizon, end - 2 * horizon
    log(f"[2/6] Building features (train snapshot {snap_train.date()}, test snapshot {snap_test.date()})")
    train_df = features.make_snapshot(sales, returns, snap_train, horizon)
    test_df = features.make_snapshot(sales, returns, snap_test, horizon)
    if len(train_df) < 200 or train_df["churn"].nunique() < 2 or test_df["churn"].nunique() < 2:
        raise RuntimeError("Not enough data/variety to train. Need more customers and a longer date range.")

    log("[3/6] Training and comparing churn models")
    res = churn_model.train_and_evaluate(train_df, test_df)
    for name, m in res["results"].items():
        log(f"      {name:20s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
    log(f"      best model: {res['best_name']}")

    log("[4/6] Segmenting customers (K-Means on RFM)")
    final = features.make_snapshot(sales, returns, end, horizon, with_label=False)
    seg = segmentation.fit_segments(final, k)

    log("[5/6] Scoring every customer")
    scores = scoring.score_customers(final, res["model"], seg)

    log("[6/6] Saving model, scores, metrics and plots")
    joblib.dump(res["model"], config.MODEL_PATH)
    joblib.dump({k_: seg[k_] for k_ in ("scaler", "kmeans", "names")}, config.SEGMENTER_PATH)
    scores.to_csv(config.SCORES_PATH)
    plots.silhouette_plot(seg["silhouette"])
    plots.segment_scatter(scores)
    plots.roc_plot(res["curves"], res["results"])
    plots.importance_plot(res["importance"])
    plots.confusion_plot(res["confusion_matrix"])

    metrics = {
        "data_source": source, "data_file": path.name,
        "raw_rows": int(len(raw)), "sales_lines": int(len(sales)),
        "customers": int(len(final)), "churn_window_days": config.CHURN_WINDOW_DAYS,
        "best_model": res["best_name"], "model_results": res["results"],
        "confusion_matrix": res["confusion_matrix"], "feature_importance": res["importance"],
        "train_rows": res["train_rows"], "test_rows": res["test_rows"],
        "train_churn_rate": res["train_churn_rate"], "test_churn_rate": res["test_churn_rate"],
        "silhouette": {str(k_): v for k_, v in seg["silhouette"].items()},
        "segment_summary": seg["summary"].reset_index().to_dict(orient="records"),
        "high_value_cutoff": float(final["monetary"].quantile(0.75)),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"Done. Scores: {config.SCORES_PATH}")
    return metrics


def artifacts_ready():
    return all(p.exists() for p in (config.MODEL_PATH, config.SEGMENTER_PATH,
                                    config.SCORES_PATH, config.METRICS_PATH))
