"""Basic checks. Run with:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_data import generate  # noqa: E402
from src import features, preprocess, scoring  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "t.csv"
        _, raw = generate(path=tmp, n_customers=300, seed=1)
        cls.raw = raw
        cls.sales, cls.returns = preprocess.clean(raw)

    def test_clean_removes_bad_rows(self):
        self.assertFalse(self.sales["CustomerID"].isna().any())
        self.assertTrue((self.sales["Quantity"] > 0).all())
        self.assertTrue((self.sales["UnitPrice"] > 0).all())
        self.assertFalse(self.sales["InvoiceNo"].str.startswith("C").any())

    def test_features_have_no_missing_values(self):
        end = self.sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
        f = features.build_features(self.sales, self.returns, end)
        self.assertEqual(list(f.columns), features.FEATURES)
        self.assertFalse(f.isna().any().any())
        self.assertTrue((f["frequency"] >= 1).all())

    def test_labels_are_binary_and_no_future_leak(self):
        end = self.sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
        snap = end - pd.Timedelta(days=90)
        df = features.make_snapshot(self.sales, self.returns, snap, pd.Timedelta(days=90))
        self.assertTrue(set(df["churn"].unique()) <= {0, 1})
        # every customer in the snapshot bought before the snapshot date
        first_orders = self.sales.groupby("CustomerID")["InvoiceDate"].min()
        self.assertTrue((first_orders.loc[df.index] < snap).all())

    def test_risk_levels(self):
        self.assertEqual(scoring.risk_level(0.1), "Low")
        self.assertEqual(scoring.risk_level(0.5), "Medium")
        self.assertEqual(scoring.risk_level(0.9), "High")

    def test_profile_row(self):
        row = features.row_from_profile(30, 6, 600, 300, 2, 15, 0)
        self.assertEqual(list(row.columns), features.FEATURES)
        self.assertAlmostEqual(row.loc[0, "avg_order_value"], 100.0)


if __name__ == "__main__":
    unittest.main()
