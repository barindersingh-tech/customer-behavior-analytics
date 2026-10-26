"""Customer-level behavioural features (RFM and more) and churn labels."""
import numpy as np
import pandas as pd

FEATURES = [
    "recency", "frequency", "monetary", "avg_order_value", "avg_gap_days",
    "recency_ratio", "tenure_days", "n_products", "orders_last_90d",
    "recent_share", "return_lines",
]

FEATURE_LABELS = {
    "recency": "Days since last order",
    "frequency": "Number of orders",
    "monetary": "Total spend",
    "avg_order_value": "Average order value",
    "avg_gap_days": "Average days between orders",
    "recency_ratio": "Recency / usual gap",
    "tenure_days": "Days since first order",
    "n_products": "Distinct products bought",
    "orders_last_90d": "Orders in last 90 days",
    "recent_share": "Share of orders in last 90 days",
    "return_lines": "Returned / cancelled lines",
}


def build_features(sales, returns, snapshot):
    """Features for every customer who bought before `snapshot` (exclusive)."""
    s = sales[sales["InvoiceDate"] < snapshot]
    orders = (s.groupby(["CustomerID", "InvoiceNo"])
                .agg(date=("InvoiceDate", "max"), amount=("Amount", "sum"))
                .reset_index())
    g = orders.groupby("CustomerID")
    last, first = g["date"].max(), g["date"].min()

    f = pd.DataFrame(index=last.index)
    f["recency"] = (snapshot - last).dt.days
    f["frequency"] = g.size()
    f["monetary"] = g["amount"].sum()
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    f["tenure_days"] = (snapshot - first).dt.days
    span = (last - first).dt.days
    gap = span / (f["frequency"] - 1).replace(0, np.nan)
    f["avg_gap_days"] = gap.fillna(f["tenure_days"] + 1)       # one-order customers
    f["recency_ratio"] = f["recency"] / f["avg_gap_days"].clip(lower=1)
    f["n_products"] = s.groupby("CustomerID")["StockCode"].nunique()

    recent = orders[orders["date"] >= snapshot - pd.Timedelta(days=90)]
    f["orders_last_90d"] = recent.groupby("CustomerID").size().reindex(f.index).fillna(0)
    f["recent_share"] = f["orders_last_90d"] / f["frequency"]

    r = returns[returns["InvoiceDate"] < snapshot]
    f["return_lines"] = r.groupby("CustomerID").size().reindex(f.index).fillna(0)

    f.index.name = "CustomerID"
    return f[FEATURES].astype(float)


def make_snapshot(sales, returns, snapshot, horizon, with_label=True):
    """Features as of `snapshot`; label = 1 if the customer did NOT buy during
    the `horizon` that follows the snapshot."""
    feat = build_features(sales, returns, snapshot)
    if with_label:
        window = sales[(sales["InvoiceDate"] >= snapshot) & (sales["InvoiceDate"] < snapshot + horizon)]
        active = set(window["CustomerID"].unique())
        feat["churn"] = (~feat.index.isin(active)).astype(int)
    return feat


def row_from_profile(recency, frequency, monetary, tenure_days, orders_last_90d, n_products, return_lines):
    """Build one feature row from a handful of user inputs (used by the
    'What-if prediction' page)."""
    frequency = max(float(frequency), 1.0)
    tenure_days = max(float(tenure_days), float(recency))
    span = max(tenure_days - float(recency), 0.0)
    avg_gap = span / (frequency - 1) if frequency > 1 else tenure_days + 1
    recent = min(float(orders_last_90d), frequency)
    row = {
        "recency": float(recency), "frequency": frequency, "monetary": float(monetary),
        "avg_order_value": float(monetary) / frequency, "avg_gap_days": avg_gap,
        "recency_ratio": float(recency) / max(avg_gap, 1.0), "tenure_days": tenure_days,
        "n_products": float(n_products), "orders_last_90d": recent,
        "recent_share": recent / frequency, "return_lines": float(return_lines),
    }
    return pd.DataFrame([row])[FEATURES]
