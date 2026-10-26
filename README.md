# Customer Behavior Analytics Using Machine Learning

A complete, runnable project that:

1. cleans retail transaction data,
2. builds customer behaviour features (RFM and more),
3. **segments customers** with K-Means (Champions, Loyal, At-Risk, Lost),
4. **predicts churn** (will the customer buy in the next 90 days?) by comparing Logistic Regression, Random Forest and Gradient Boosting on a later time period,
5. adds **rule-based checks** (reasons, risk level, priority, suggested action),
6. shows everything in a **Flask web dashboard**.

## Quick start (VS Code)

1. Unzip the folder and open it in VS Code (`File > Open Folder`).
2. Open a terminal (`Ctrl + ~`) and create a virtual environment:

   **Windows**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   **macOS / Linux**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   python app.py
   ```
   The first run creates demo data and trains the models (about 20-60 seconds).
   Then open **http://127.0.0.1:5000** in your browser.

You can also press **F5** in VS Code and choose "Run dashboard (app.py)".

Other commands:

| Command | What it does |
|---|---|
| `python train.py` | Re-train everything and rewrite `models/`, `outputs/`, `static/plots/` |
| `python train.py --k 5` | Use 5 customer segments instead of 4 |
| `python generate_data.py` | Re-create the synthetic demo data |
| `python -m unittest discover -s tests -v` | Run the tests |

## Important: demo data vs real data

By default the project uses **synthetic (artificial) data** created by `generate_data.py`, so it runs with no downloads. Numbers produced from it are fine for testing the project, but **they are not real-world findings**. Do not present them as results from a real company in your report or viva.

To use the real **UCI Online Retail** dataset:

1. Download it from the UCI Machine Learning Repository (search "Online Retail UCI", dataset 352) or from Kaggle.
2. Put the file in the `data/` folder named `Online Retail.xlsx` (or save it as `online_retail.csv`).
3. Delete `models/model.pkl` (or just run) `python train.py`.

The project automatically prefers a real file in `data/` over the synthetic one. The dashboard header shows which source is in use. Your real metrics are saved in `outputs/metrics.json` (copy them into the results table of your report).

## How it works

**Cleaning** (`src/preprocess.py`): drops rows without a customer ID, cancelled invoices (starting with "C"), and zero/negative quantities or prices.

**Features** (`src/features.py`): recency, frequency, monetary value, average order value, average days between orders, recency / usual gap, tenure, distinct products, orders in last 90 days, share of recent orders, returned lines.

**Churn label and time-based split** (`src/pipeline.py`): features are computed as of a *snapshot date*; the label is 1 if the customer made **no purchase in the 90 days after** the snapshot. Models are trained on the earlier snapshot (end - 180 days) and tested on the later one (end - 90 days), so the test mimics predicting the future and there is no leakage.

**Segments** (`src/segmentation.py`): K-Means on log-scaled, standardised recency / frequency / monetary. Clusters are named by ranking `z(frequency) + z(monetary) - z(recency)`. The silhouette plot helps you justify the choice of k.

**Rules** (`src/scoring.py`):

| Rule | Effect |
|---|---|
| R1 last order more than 2x the usual gap ago | adds a reason |
| R2 no orders in last 90 days despite 3+ past orders | adds a reason |
| R3 only one order so far | adds a reason |
| R4 3+ returned/cancelled lines | adds a reason |
| R5 Low risk but silence is more than 3x the usual gap | risk raised to Medium |
| R6 High risk and spend in top 25% | priority "Contact first" |

Risk level: probability below 0.33 = Low, 0.33 to 0.66 = Medium, 0.66 and above = High (change in `config.py`).

## Dashboard pages

* **Dashboard** - KPIs, segment and risk charts, segment table, biggest high-risk customers, model comparison, plots.
* **Customer page** (`/customer/<id>`) - segment, churn probability, risk, reasons, suggested action, features.
* **What-if prediction** (`/predict`) - type a customer profile and get the risk.
* **History** (`/history`) - customers you looked up.
* **API** - `/api/customer/<id>` and `/api/summary` return JSON.
* **Download** - all scores as CSV.

## Folder structure

```
app.py               Flask dashboard
train.py             command-line training
generate_data.py     synthetic demo data
config.py            settings (paths, churn window, clusters, thresholds)
src/                 preprocess, features, segmentation, churn_model, scoring, plots, pipeline
templates/ static/   web pages, CSS, generated plots
data/                put the real dataset here
models/              saved model.pkl and segmenter.pkl (created by training)
outputs/             customer_scores.csv, metrics.json, history.json
tests/               unit tests
```

## Troubleshooting

* `ModuleNotFoundError` - the virtual environment is not active, or step 3 was skipped.
* Port 5000 already in use - change `port=5000` at the bottom of `app.py`.
* Charts on the dashboard are blank - they load Chart.js from the internet; check your connection.
* Real dataset error "missing columns" - the file must have `InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID` columns.

## Limits

Churn is defined by a 90-day silence, which is a business choice. The model uses only purchase behaviour, not demographics or marketing history, and should be retrained regularly. Segment names come from a simple ranking rule and should be checked by a person.
