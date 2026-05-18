"""
preprocessing.py
----------------
Loads creditcard.csv, scales the two raw-valued columns (Time and Amount),
splits into train / test sets, and saves everything to disk.

Run this before any model training script.

Usage
-----
    python preprocessing.py
    python preprocessing.py --test_size 0.2 --random_state 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
SPLIT_DIR = ROOT / "data" / "splits"   # where the .npy arrays are saved


# ---------------------------------------------------------------------------
# Step 1 — Load the raw CSV
# ---------------------------------------------------------------------------

def load_csv(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path.name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Quick sanity check — confirm expected columns are present
    expected = {"Time", "Amount", "Class"}
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Report class balance so we can see the imbalance clearly
    n_fraud  = (df["Class"] == 1).sum()
    n_legit  = (df["Class"] == 0).sum()
    ratio    = n_legit / n_fraud
    print(f"  Legitimate transactions : {n_legit:,}  ({n_legit/len(df):.2%})")
    print(f"  Fraud transactions      : {n_fraud:,}  ({n_fraud/len(df):.2%})")
    print(f"  Imbalance ratio         : {ratio:.0f}:1  (legitimate:fraud)")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Separate features from label
# ---------------------------------------------------------------------------

def split_features_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    X contains all columns except Class.
    y contains only the Class column (0 = legit, 1 = fraud).

    Column order in X:
        index 0  → Time
        index 1  → V1
        ...
        index 28 → V28
        index 29 → Amount
    """
    X = df.drop(columns=["Class"]).values   # shape: (284807, 30)
    y = df["Class"].values                  # shape: (284807,)
    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Label vector shape   : {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Step 3 — Train / test split
# ---------------------------------------------------------------------------

def make_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    stratify=y ensures both splits keep the same fraud rate (~0.17%).
    Without it a random split might put almost no fraud cases in the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\nTrain set : {X_train.shape[0]:,} rows  "
          f"({y_train.mean():.3%} fraud)")
    print(f"Test set  : {X_test.shape[0]:,} rows  "
          f"({y_test.mean():.3%} fraud)")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 4 — Scale Time and Amount
# ---------------------------------------------------------------------------

# Indices of the two columns that need scaling.
# V1–V28 are already on a small scale because PCA normalises them.
TIME_COL   = 0   # first column
AMOUNT_COL = 29  # last column

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the training data only, then apply it to both
    splits.

    Why fit on training only?
        If we fit on the full dataset (or on test data), the scaler learns
        the test set's mean and std.  That information then leaks into the
        model indirectly.  It's a subtle form of cheating known as data
        leakage.  Small effect on a large dataset like this, but always the
        right habit.

    Why only Time and Amount?
        V1–V28 are PCA components — PCA already centres and scales them.
        Scaling them again would do nothing harmful, but it's unnecessary.

    After scaling, Time and Amount will have:
        mean  ≈ 0
        std   ≈ 1
    """
    scaler = StandardScaler()

    # fit_transform on train: learns mean/std AND applies the transformation
    X_train[:, [TIME_COL, AMOUNT_COL]] = scaler.fit_transform(
        X_train[:, [TIME_COL, AMOUNT_COL]]
    )

    # transform on test: applies the SAME mean/std learned from train
    # (do NOT call fit_transform here)
    X_test[:, [TIME_COL, AMOUNT_COL]] = scaler.transform(
        X_test[:, [TIME_COL, AMOUNT_COL]]
    )

    print(f"\nScaling applied to columns: Time (idx {TIME_COL}), "
          f"Amount (idx {AMOUNT_COL})")
    print(f"  Time   — mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f}")
    print(f"  Amount — mean: {scaler.mean_[1]:.2f}, std: {scaler.scale_[1]:.2f}")

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Step 5 — Save everything to disk
# ---------------------------------------------------------------------------

def save_splits(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    out_dir: Path,
) -> None:
    """
    Persist the processed arrays and fitted scaler.

    Files written:
        X_train.npy  — training features, scaled
        X_test.npy   — test features, scaled
        y_train.npy  — training labels
        y_test.npy   — test labels
        scaler.pkl   — fitted StandardScaler (needed at inference time)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy",  y_test)

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved to {out_dir}/")
    for name in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "scaler.pkl"]:
        size = (out_dir / name).stat().st_size / 1024
        print(f"  {name:<14} {size:>8.1f} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path, out_dir: Path, test_size: float, random_state: int) -> None:
    print("=" * 50)
    print("Step 1 — Load CSV")
    print("=" * 50)
    df = load_csv(data_dir)

    print("\n" + "=" * 50)
    print("Step 2 — Separate features and label")
    print("=" * 50)
    X, y = split_features_label(df)

    print("\n" + "=" * 50)
    print("Step 3 — Train / test split")
    print("=" * 50)
    X_train, X_test, y_train, y_test = make_split(X, y, test_size, random_state)

    print("\n" + "=" * 50)
    print("Step 4 — Scale Time and Amount")
    print("=" * 50)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    print("\n" + "=" * 50)
    print("Step 5 — Save to disk")
    print("=" * 50)
    save_splits(X_train, X_test, y_train, y_test, scaler, out_dir)

    print("\nPreprocessing complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess the credit card fraud dataset")
    p.add_argument("--test_size",    type=float, default=0.2)
    p.add_argument("--random_state", type=int,   default=42)
    p.add_argument("--data_dir",     type=Path,  default=DATA_DIR)
    p.add_argument("--out_dir",      type=Path,  default=SPLIT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.out_dir, args.test_size, args.random_state)