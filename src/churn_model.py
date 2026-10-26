"""Train and compare churn prediction models."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config
from src.features import FEATURES


def candidate_models():
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=rs)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=rs),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200, random_state=rs),
    }


def _metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def train_and_evaluate(train_df, test_df):
    """Fit every candidate on the earlier snapshot, evaluate on the later one,
    and return the best model (highest ROC-AUC)."""
    Xtr, ytr = train_df[FEATURES], train_df["churn"]
    Xte, yte = test_df[FEATURES], test_df["churn"]

    results, fitted, curves = {}, {}, {}
    for name, model in candidate_models().items():
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        results[name] = _metrics(yte, prob)
        fitted[name] = model
        fpr, tpr, _ = roc_curve(yte, prob)
        curves[name] = (fpr, tpr)

    best = max(results, key=lambda n: results[n]["roc_auc"])
    best_prob = fitted[best].predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, (best_prob >= 0.5).astype(int)).tolist()

    imp = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=5,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()),
                             key=lambda kv: kv[1], reverse=True))
    return {
        "model": fitted[best], "best_name": best, "results": results, "curves": curves,
        "confusion_matrix": cm, "importance": importance,
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "train_churn_rate": float(ytr.mean()), "test_churn_rate": float(yte.mean()),
    }
