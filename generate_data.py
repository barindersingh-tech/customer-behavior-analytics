"""Create a synthetic retail transaction file with the same columns as the UCI
"Online Retail" dataset (InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice, CustomerID, Country).

The data is artificial. It exists so the project runs out of the box.
For real results, put the real dataset in the data/ folder (see README).

Usage:  python generate_data.py
"""
import numpy as np
import pandas as pd

import config

ADJ = ["Vintage", "Classic", "Modern", "Rustic", "Cozy", "Bright", "Ceramic", "Wooden", "Glass", "Floral"]
NOUN = ["Mug", "Lamp", "Candle", "Frame", "Cushion", "Teapot", "Bowl", "Clock", "Vase", "Tray", "Basket", "Jar"]
COUNTRIES = ["United Kingdom", "Germany", "France", "Spain", "Netherlands", "Ireland", "Belgium"]
COUNTRY_P = [0.84, 0.05, 0.04, 0.02, 0.02, 0.02, 0.01]

# name: (share of customers, orders per month, avg lines per order, qty scale, chance of eventually churning)
CUSTOMER_TYPES = {
    "champion": (0.08, 3.0, 8, 3.0, 0.05),
    "loyal": (0.22, 1.5, 5, 2.0, 0.12),
    "occasional": (0.35, 0.6, 4, 1.5, 0.35),
    "new_or_oneoff": (0.20, 0.25, 3, 1.2, 0.60),
    "lapsing": (0.15, 1.0, 4, 1.5, 0.90),
}


def generate(path=None, n_customers=2500, seed=42, start="2025-01-01", end="2026-09-30"):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_days = (end - start).days

    # product catalogue (some products are much more popular than others)
    n_products = 120
    catalogue = pd.DataFrame({
        "StockCode": [f"P{1000 + i}" for i in range(n_products)],
        "Description": [f"{rng.choice(ADJ)} {rng.choice(NOUN)}" for _ in range(n_products)],
        "UnitPrice": np.clip(np.round(np.exp(rng.normal(1.8, 0.7, n_products)), 2), 0.5, 120),
    })
    popularity = rng.zipf(1.6, n_products).astype(float)
    popularity /= popularity.sum()

    names = list(CUSTOMER_TYPES)
    shares = np.array([CUSTOMER_TYPES[n][0] for n in names])
    types = rng.choice(names, size=n_customers, p=shares / shares.sum())

    rows, invoice_no = [], 540000
    for i, ctype in enumerate(types):
        _, rate, lines_mean, qty_scale, p_churn = CUSTOMER_TYPES[ctype]
        cust_id = 10000 + i
        country = rng.choice(COUNTRIES, p=COUNTRY_P)
        first = start + pd.Timedelta(days=float(total_days * 0.92 * rng.beta(1, 2.0)))
        churn_date = None
        if rng.random() < p_churn:
            churn_date = first + pd.Timedelta(days=float(rng.uniform(30, 330)))
        mean_gap = 30.0 / rate

        t = first
        while t <= end:
            if churn_date is not None and t > churn_date:
                break
            invoice_no += 1
            # customers about to churn buy less and less often
            slowdown = 1.0
            if churn_date is not None:
                slowdown += 1.4 * max(0.0, (t - (churn_date - pd.Timedelta(days=90))).days / 90.0)
            hour = int(rng.integers(8, 19))
            stamp = t.normalize() + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            n_lines = 1 + rng.poisson(lines_mean - 1)
            items = rng.choice(n_products, size=n_lines, replace=False, p=popularity) if n_lines <= n_products else []
            guest = rng.random() < 0.04  # some orders have no customer id (guest checkout)
            for p in items:
                qty = 1 + rng.poisson(qty_scale)
                rows.append((str(invoice_no), catalogue.StockCode[p], catalogue.Description[p], qty,
                             stamp, catalogue.UnitPrice[p], np.nan if guest else cust_id, country))
            # a small share of orders is partly cancelled the next day
            if rng.random() < 0.025 and len(items) > 0:
                invoice_no += 1
                p = items[0]
                rows.append(("C" + str(invoice_no), catalogue.StockCode[p], catalogue.Description[p], -1,
                             stamp + pd.Timedelta(days=1), catalogue.UnitPrice[p],
                             np.nan if guest else cust_id, country))
            t = t + pd.Timedelta(days=float(max(1.0, rng.gamma(2.0, mean_gap / 2.0) * slowdown)))

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df = df[df.InvoiceDate <= end + pd.Timedelta(days=1)].sort_values("InvoiceDate").reset_index(drop=True)
    path = path or config.SYNTHETIC_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path, df


if __name__ == "__main__":
    p, d = generate()
    print(f"Wrote {len(d):,} rows for {d.CustomerID.nunique():,} customers to {p}")
