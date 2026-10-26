"""Central settings for the Customer Behavior Analytics project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "static" / "plots"

# If one of these files exists it is used (real UCI "Online Retail" data).
# Otherwise the project falls back to the synthetic demo data below.
REAL_DATA_CANDIDATES = [
    DATA_DIR / "online_retail.csv",
    DATA_DIR / "Online Retail.xlsx",
    DATA_DIR / "online_retail.xlsx",
    DATA_DIR / "Online_Retail.xlsx",
]
SYNTHETIC_PATH = DATA_DIR / "transactions.csv"

MODEL_PATH = MODEL_DIR / "model.pkl"
SEGMENTER_PATH = MODEL_DIR / "segmenter.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SCORES_PATH = OUTPUT_DIR / "customer_scores.csv"
HISTORY_PATH = OUTPUT_DIR / "history.json"

CHURN_WINDOW_DAYS = 90      # a customer "churns" if they do not buy in this many days
N_CLUSTERS = 4              # number of customer segments (K-Means)
RANDOM_STATE = 42
RISK_LOW = 0.33             # churn probability below this  -> Low risk
RISK_HIGH = 0.66            # churn probability at/above this -> High risk
