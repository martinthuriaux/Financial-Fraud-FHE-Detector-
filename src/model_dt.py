"""
model_dt.py
-----------
Trains a quantisation-aware Decision Tree classifier with Concrete-ML,
compiles it to an FHE circuit, and saves all artefacts.

A single Decision Tree is the simplest FHE-compatible model in this
project.  It produces the smallest circuit, the fastest FHE inference,
and is fully human-readable — every split can be printed and inspected.

The tradeoff: lower accuracy than XGBoost (which uses 50+ trees) but
much faster FHE inference and complete interpretability.

Dependency: preprocessing.py must have run first.

Usage
-----
    python model_dt.py                        # default settings
    python model_dt.py --max_depth 6          # specific tree depth
    python model_dt.py --depth_sweep          # compare depths 3-8
    python model_dt.py --skip_compile         # train + evaluate only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.tree import export_text
from concrete.ml.sklearn import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Section 1 - Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts" / "decision_tree"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Section 2 - Load splits
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path) -> tuple[np.ndarray, ...]:
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    X_test  = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    y_test  = np.load(data_dir / "y_test.npy").astype(np.int64)

    print(f"Train : {X_train.shape}  fraud rate: {(y_train == 1).mean():.3%}")
    print(f"Test  : {X_test.shape}   fraud rate: {(y_test == 1).mean():.3%}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Section 3 - Class weight
# ---------------------------------------------------------------------------

def compute_class_weight(y_train: np.ndarray) -> dict:
    """
    Return per-class weights using the same sqrt formula as the XGBoost model.

    "balanced" (the sklearn shorthand) uses the raw ratio (~577), which is
    too aggressive and collapses precision.  sqrt(n_legit / n_fraud) (~24)
    moderates the signal while still strongly up-weighting fraud.
    """
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    w_fraud = (n_legit / n_fraud) ** 0.5
    print(f"\nClass weight — legit: 1.00,  fraud: {w_fraud:.2f}")
    return {0: 1.0, 1: w_fraud}


# ---------------------------------------------------------------------------
# Section 4 - Build and train
# ---------------------------------------------------------------------------

def build_model(
    n_bits:           int,
    max_depth:        int,
    class_weight:     dict  | None = None,
    min_samples_leaf: int          = 5,
    criterion:        str          = "entropy",
) -> DecisionTreeClassifier:
    """
    Construct the Concrete-ML DecisionTreeClassifier.

    class_weight : dict
        Per-class weights passed to the split criterion.  Uses the same
        sqrt(n_legit / n_fraud) formula as the XGBoost model (~24) rather
        than sklearn's "balanced" shorthand which gives the raw ratio (~577)
        and pushes recall too high at the expense of precision.

    min_samples_leaf : int
        Minimum number of training samples required at a leaf node.
        Default sklearn value is 1 (a leaf can form on a single sample).
        Setting to 5 prevents the tree from memorising individual fraud
        cases, similar to min_child_weight in XGBoost.

    criterion : str
        The function used to measure split quality.  "entropy" (information
        gain) is more sensitive to small impurity changes than "gini",
        which can help find better splits in a heavily imbalanced dataset.

    max_depth
        The only meaningful capacity parameter for a single tree.
        Controls the maximum number of nested splits from root to leaf.

        depth 3 →  8 possible leaves, very simple rules
        depth 5 →  32 possible leaves, moderate complexity
        depth 7 →  128 possible leaves, captures fine-grained patterns
        depth None → grows until pure leaves, almost always overfits

        For FHE: deeper trees produce larger circuits and slower inference.
        depth 5-6 is the practical sweet spot for this dataset.

    n_bits
        Quantisation precision.  Decision Trees are the most tolerant
        of low n_bits among all three models because their computation
        is just integer comparisons — no matrix multiplications that
        accumulate rounding error across many operations.
        n_bits=6 is usually sufficient with negligible accuracy loss.
    """
    model = DecisionTreeClassifier(
        n_bits=n_bits,
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42,
    )
    return model


def train(
    model:   DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> DecisionTreeClassifier:
    """
    Train in plaintext.

    Decision Tree training is extremely fast — under a second — because
    it makes a single greedy pass through the data, finding the best
    split at each node without iteration.
    """
    print("\n--- Training Decision Tree ---")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.3f}s")
    print(f"Tree depth  : {model.get_depth()}")
    print(f"Tree leaves : {model.get_n_leaves()}")
    return model


# ---------------------------------------------------------------------------
# Section 4 - Print tree (interpretability)
# ---------------------------------------------------------------------------

# Feature names matching the column order from preprocessing.py
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def print_tree(model: DecisionTreeClassifier, max_depth: int =6) -> None:
    """
    Print a human-readable version of the decision tree.

    This is only meaningful for shallow trees (max_depth <= 5).
    For deeper trees it becomes too large to read.

    The output shows exactly which features and thresholds the model
    uses at each split — something impossible with XGBoost or the NN.

    Example output:
        |--- V14 <= -2.50
        |   |--- V10 <= -3.12
        |   |   |--- class: 1  (fraud)
        |   |--- V10 > -3.12
        |   |   |--- class: 0  (legit)
        |--- V14 > -2.50
        |   |--- class: 0  (legit)
    """
    if max_depth > 6:
        print("\n(Tree too deep to print readably — skipping)")
        return

    print(f"\n--- Decision Tree structure (first {max_depth} levels) ---")

    # Concrete-ML wraps the sklearn tree — access it via .sklearn_model
    sklearn_tree = model.sklearn_model
    tree_text = export_text(
        sklearn_tree,
        feature_names=FEATURE_NAMES,
        max_depth=3,           # only show top 3 levels for readability
        show_weights=False,
    )
    print(tree_text)


# ---------------------------------------------------------------------------
# Section 5 - Plaintext evaluation with threshold tuning
# ---------------------------------------------------------------------------

def evaluate_plaintext(
    model:  DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    print("\n--- Plaintext evaluation ---")
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores      = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx       = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx])

    print(f"Default threshold : 0.50")
    print(f"Optimal threshold : {best_threshold:.4f}  "
          f"(precision={prec[best_idx]:.2f}, recall={rec[best_idx]:.2f})")

    y_pred_default = (y_proba >= 0.50).astype(int)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    print("\n-- With default threshold (0.50) --")
    print(classification_report(y_test, y_pred_default,
                                target_names=["legit", "fraud"]))

    print(f"-- With optimal threshold ({best_threshold:.4f}) --")
    print(classification_report(y_test, y_pred_optimal,
                                target_names=["legit", "fraud"]))

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}  <- primary metric under class imbalance")

    return {
        "roc_auc":        roc_auc,
        "pr_auc":         pr_auc,
        "best_threshold": best_threshold,
    }


# ---------------------------------------------------------------------------
# Section 6 - max_depth sweep
# ---------------------------------------------------------------------------

def depth_sweep(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    n_bits:     int = 6,
    depth_range: list | None = None,
) -> None:
    """
    Train and evaluate at several max_depth values without FHE compilation.

    Unlike XGBoost where n_estimators and max_depth both matter, for a
    single Decision Tree max_depth is the primary knob.

    Use this to find the depth that best balances accuracy vs FHE circuit
    size before committing to compilation.

    Typical finding on this dataset:
        depth 3-4  → underfits, misses complex fraud patterns
        depth 5-6  → good balance, clean FHE circuit
        depth 7+   → overfits, marginal accuracy gain, larger circuit
    """
    if depth_range is None:
        depth_range = [3, 4, 5, 6, 7, 8]

    print("\n--- max_depth sweep ---")
    print(f"{'depth':<8} {'ROC-AUC':<10} {'PR-AUC':<10} "
          f"{'Leaves':<8} {'Train time'}")
    print("-" * 50)

    cw = compute_class_weight(y_train)
    for depth in depth_range:
        model = build_model(n_bits=n_bits, max_depth=depth, class_weight=cw)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)
        leaves  = model.get_n_leaves()

        print(f"{depth:<8} {roc_auc:<10.4f} {pr_auc:<10.4f} "
              f"{leaves:<8} {elapsed:.3f}s")

    print("\nNote: more leaves = larger FHE circuit = slower inference.")


# ---------------------------------------------------------------------------
# Section 7 - FHE compilation and simulation evaluation
# ---------------------------------------------------------------------------

def compile_to_fhe(
    model:   DecisionTreeClassifier,
    X_train: np.ndarray,
):
    """
    Compile to FHE circuit.

    A Decision Tree compiles to the simplest circuit of all three models.
    Each prediction follows one path from root to leaf — a sequence of
    integer comparisons with no accumulation across multiple trees.

    Expect compilation in under 30 seconds and FHE simulation latency
    significantly faster than XGBoost.
    """
    print("\n--- Compiling to FHE circuit ---")
    t0 = time.perf_counter()
    fhe_circuit = model.compile(X_train)
    elapsed = time.perf_counter() - t0
    print(f"Compilation complete in {elapsed:.1f}s")
    return fhe_circuit


def evaluate_fhe(
    model:     DecisionTreeClassifier,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    n_samples: int = 100,
) -> dict:
    print(f"\n--- FHE simulation ({n_samples} samples) ---")
    rng       = np.random.default_rng(42)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_fraud   = min(len(fraud_idx), n_samples // 2)
    n_legit   = n_samples - n_fraud
    idx = np.concatenate([
        rng.choice(fraud_idx, n_fraud, replace=False),
        rng.choice(legit_idx, n_legit, replace=False),
    ])
    rng.shuffle(idx)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    t0      = time.perf_counter()
    y_pred  = model.predict(X_sub, fhe="simulate")
    elapsed = time.perf_counter() - t0

    roc_auc    = roc_auc_score(y_sub, y_pred)
    per_sample = elapsed / n_samples

    print(classification_report(y_sub, y_pred,
                                target_names=["legit", "fraud"]))
    print(f"ROC-AUC (FHE sim) : {roc_auc:.4f}")
    print(f"Latency           : {per_sample * 1000:.1f} ms / sample")

    return {"roc_auc_fhe": roc_auc, "latency_ms": per_sample * 1000}


# ---------------------------------------------------------------------------
# Section 8 - Save artefacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model:         DecisionTreeClassifier,
    params:        dict,
    metrics:       dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # FHE circuit serialization via FHEModelDev triggers a libc++ crash on macOS
    # (concrete-python LLVM bug). The circuit is compiled at server startup instead
    # — compile once, serve forever — so no disk serialization is needed here.

    # Plaintext sklearn model for fast non-encrypted inference.
    joblib.dump(model.sklearn_model, artifacts_dir / "sklearn_model.joblib")

    with open(artifacts_dir / "threshold.json", "w") as f:
        json.dump({"threshold": metrics.get("best_threshold", 0.5)}, f)

    metrics_to_save = {k: v for k, v in metrics.items()
                       if k != "best_threshold"}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    with open(artifacts_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nArtefacts saved to {artifacts_dir}")
    for p in sorted(artifacts_dir.iterdir()):
        print(f"  {p.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and compile Decision Tree FHE fraud detector"
    )
    p.add_argument("--n_bits",            type=int,   default=6)
    p.add_argument("--max_depth",         type=int,   default=6)
    p.add_argument("--min_samples_leaf",  type=int,   default=5)
    p.add_argument("--criterion",         type=str,   default="entropy")
    p.add_argument("--depth_sweep",  action="store_true",
                   help="Sweep max_depth 3-8 and exit")
    p.add_argument("--skip_compile", action="store_true",
                   help="Skip FHE compilation")
    p.add_argument("--data_dir",     type=Path, default=DATA_DIR)
    p.add_argument("--artifacts_dir",type=Path, default=ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test = load_splits(args.data_dir)
    cw = compute_class_weight(y_train)

    if args.depth_sweep:
        depth_sweep(X_train, y_train, X_test, y_test, n_bits=args.n_bits)
        return

    params = dict(
        n_bits=args.n_bits,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
    )
    model = build_model(class_weight=cw, **params)
    model = train(model, X_train, y_train)

    # Print tree structure — unique to this model
    print_tree(model, max_depth=args.max_depth)

    pt_metrics = evaluate_plaintext(model, X_test, y_test)

    if args.skip_compile:
        print("\nSkipping FHE compilation (--skip_compile set).")
        return

    compile_to_fhe(model, X_train)
    fhe_metrics = evaluate_fhe(model, X_test, y_test)

    delta = abs(pt_metrics["roc_auc"] - fhe_metrics["roc_auc_fhe"])
    print(f"\nROC-AUC delta (plaintext vs FHE sim): {delta:.4f}")
    if delta > 0.02:
        print("  Warning: delta > 0.02 - consider increasing n_bits.")

    all_metrics = {**pt_metrics, **fhe_metrics}
    save_artifacts(model, params, all_metrics, args.artifacts_dir)


if __name__ == "__main__":
    main()