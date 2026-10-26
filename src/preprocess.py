"""Loading and cleaning of the raw transaction data."""
import pandas as pd

import config

REQUIRED = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]


def resolve_data_path():
    """Return (path, source) where source is 'real' or 'synthetic'.
    Creates the synthetic file if no data exists yet."""
    for p in config.REAL_DATA_CANDIDATES:
        if p.exists():
            return p, "real"
    if not config.SYNTHETIC_PATH.exists():
        from generate_data import generate
        generate()
    return config.SYNTHETIC_PATH, "synthetic"


def load_raw(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="latin-1" if "retail" in str(path).lower() else None)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")
    return df


def clean(raw):
    """Return (sales, returns).

    sales   - valid purchase lines with a customer id and an Amount column
    returns - cancelled / returned lines (InvoiceNo starting with 'C')
    """
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)
    df = df.drop_duplicates()

    cancelled = df["InvoiceNo"].str.upper().str.startswith("C")
    returns = df[cancelled].copy()
    sales = df[~cancelled & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    sales["Amount"] = sales["Quantity"] * sales["UnitPrice"]
    return sales.reset_index(drop=True), returns.reset_index(drop=True)
