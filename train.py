"""Train the models and write all outputs.

    python train.py            # uses real data if present in data/, else synthetic
    python train.py --k 5      # use 5 customer segments
"""
import argparse

from src.pipeline import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="number of customer segments (default 4)")
    args = ap.parse_args()
    run_training(k=args.k)
