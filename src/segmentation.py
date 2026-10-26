"""Customer segmentation with K-Means on log-scaled RFM values."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config

RFM = ["recency", "frequency", "monetary"]

SEGMENT_ACTIONS = {
    "Champions": "Reward them: early access, loyalty perks, ask for reviews or referrals.",
    "Loyal Customers": "Keep them engaged: loyalty points, cross-selling and bundle offers.",
    "At-Risk Customers": "Win them back: a reminder or personal offer before they drift away.",
    "Lost Customers": "Low-cost win-back campaign; do not over-invest.",
}


def _transform(df):
    return np.log1p(df[RFM].clip(lower=0))


def _names_for(k):
    if k == 3:
        return ["Champions", "At-Risk Customers", "Lost Customers"]
    base = ["Champions", "Loyal Customers"] + ["At-Risk Customers"] * (k - 3) + ["Lost Customers"]
    return base


def name_clusters(centers):
    """Rank clusters by a simple value score and give them readable names.
    score = z(frequency) + z(monetary) - z(recency)   (best customers first)"""
    z = (centers - centers.mean()) / centers.std(ddof=0).replace(0, 1)
    score = z["frequency"] + z["monetary"] - z["recency"]
    ranked = score.sort_values(ascending=False).index.tolist()
    names = _names_for(len(ranked))
    return {cluster: names[i] for i, cluster in enumerate(ranked)}


def fit_segments(final_df, k=None):
    k = k or config.N_CLUSTERS
    X = _transform(final_df)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Xs)

    centers = final_df.assign(cluster=km.labels_).groupby("cluster")[RFM].mean()
    names = name_clusters(centers)

    # silhouette score for several k (on a sample, to stay fast)
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(4000, len(Xs)), replace=False)
    sil = {}
    for kk in range(2, 9):
        lab = KMeans(n_clusters=kk, n_init=5, random_state=config.RANDOM_STATE).fit_predict(Xs[idx])
        sil[kk] = float(silhouette_score(Xs[idx], lab))

    summary = (final_df.assign(segment=pd.Series(km.labels_, index=final_df.index).map(names))
               .groupby("segment")
               .agg(customers=("recency", "size"), avg_recency=("recency", "mean"),
                    avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"))
               .round(1))
    return {"scaler": scaler, "kmeans": km, "names": names, "centers": centers,
            "silhouette": sil, "summary": summary, "k": k}


def assign(seg, df):
    labels = seg["kmeans"].predict(seg["scaler"].transform(_transform(df)))
    return pd.Series(labels, index=df.index).map(seg["names"])
