"""Turn model output into a risk level, reasons and a suggested action
(the 'rule-based analysis' layer)."""
import numpy as np
import pandas as pd

import config
from src import segmentation
from src.features import FEATURES


def risk_level(prob):
    if prob >= config.RISK_HIGH:
        return "High"
    if prob >= config.RISK_LOW:
        return "Medium"
    return "Low"


def apply_rules(row, high_value_cutoff):
    """Return (risk_level, priority, reasons) for one customer row.

    Rules (documented in the README):
      R1  not bought for more than 2x the usual gap        -> reason
      R2  no orders in the last 90 days despite >= 3 past  -> reason
      R3  only one order so far                            -> reason
      R4  3 or more returned/cancelled lines               -> reason
      R5  recency > 3x usual gap and level is Low          -> raise level to Medium
      R6  High risk and high spend                         -> priority 'Contact first'
    """
    level = risk_level(row["churn_probability"])
    reasons = []
    if row["frequency"] > 1 and row["recency_ratio"] > 2:
        reasons.append(f"Last order was {row['recency']:.0f} days ago, over twice their usual "
                       f"gap of {row['avg_gap_days']:.0f} days")
    if row["orders_last_90d"] == 0 and row["frequency"] >= 3:
        reasons.append(f"No orders in the last 90 days despite {row['frequency']:.0f} past orders")
    if row["frequency"] == 1:
        reasons.append("Only one order so far - no repeat purchase yet")
    if row["return_lines"] >= 3:
        reasons.append(f"{row['return_lines']:.0f} returned or cancelled lines")
    if level == "Low" and row["frequency"] > 1 and row["recency_ratio"] > 3:
        level = "Medium"
        reasons.append("Rule: long silence compared with normal pattern, risk raised to Medium")
    priority = "Contact first" if (level == "High" and row["monetary"] >= high_value_cutoff) else "Normal"
    if priority == "Contact first":
        reasons.append("High-value customer at high risk - prioritise retention")
    if not reasons:
        reasons.append("Recent and regular buying pattern")
    return level, priority, reasons


def suggest_action(segment, level):
    base = segmentation.SEGMENT_ACTIONS.get(segment, "Monitor this customer.")
    if level == "High" and segment != "Lost Customers":
        return "Urgent: " + base
    return base


def score_customers(final_df, model, seg):
    """Full scoring table for every customer in `final_df`."""
    out = final_df[FEATURES].copy()
    out["segment"] = segmentation.assign(seg, final_df)
    out["churn_probability"] = model.predict_proba(final_df[FEATURES])[:, 1]
    cutoff = final_df["monetary"].quantile(0.75)

    levels, priorities, reasons = [], [], []
    for _, row in out.iterrows():
        lv, pr, rs = apply_rules(row, cutoff)
        levels.append(lv)
        priorities.append(pr)
        reasons.append(" | ".join(rs))
    out["risk_level"] = levels
    out["priority"] = priorities
    out["risk_score"] = (out["churn_probability"] * 100).round().astype(int)
    out["reasons"] = reasons
    out["suggested_action"] = [suggest_action(s, l) for s, l in zip(out["segment"], out["risk_level"])]
    out["churn_probability"] = out["churn_probability"].round(4)
    return out


def score_profile(model, row_df, high_value_cutoff):
    """Score a single hand-entered profile (What-if page)."""
    prob = float(model.predict_proba(row_df[FEATURES])[:, 1][0])
    row = row_df.iloc[0].to_dict()
    row["churn_probability"] = prob
    level, priority, reasons = apply_rules(row, high_value_cutoff)
    return {"probability": prob, "risk_score": int(round(prob * 100)), "risk_level": level,
            "priority": priority, "reasons": reasons}
