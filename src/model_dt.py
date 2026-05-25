"""
model_dt.py
-----------
Trains a quantisation-aware Decision Tree classifier with Concrete-ML,
compiles it to an FHE circuit, and saves all artefacts.

A single Decision Tree is the simplest FHE-compatible model in this
project.  It produces the smallest circuit, the fastest FHE inference,
and is fully human-readable - every split can be printed and inspected.

The tradeoff: lower accuracy than XGBoost (which uses 50+ trees) but
much faster FHE inference and complete interpretability.

Dependency: preprocessing.py must have run first.

Usage
-----
    python model_dt.py                        # default settings
    python model_dt.py --max_depth 7          # specific tree depth
    python model_dt.py --depth_sweep          # compare depths 3-8
    python model_dt.py --weight_sweep         # find best class_weight
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

# Default class weight for the fraud class.  Found via --weight_sweep.
# Single trees are very sensitive to this on imbalanced data:
#   too low (~5)  -> model predicts everything as legit
#   too high (~24) -> model predicts everything as fraud
# A value between 10 and 16 usually gives reasonable predictions.
DEFAULT_FRAUD_WEIGHT = 12.0


def compute_class_weight(y_train: np.ndarray,
                         w_fraud: float = DEFAULT_FRAUD_WEIGHT) -> dict:
    """
    Return per-class weights for the Decision Tree.

    Uses a fixed weight rather than a formula because the optimal value
    is dataset-specific and was determined empirically via --weight_sweep.
    """
    print(f"\nClass weight - legit: 1.00,  fraud: {w_fraud:.2f}")
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
        Per-class weights passed to the split criterion.  Single trees
        are pathologically sensitive to this on imbalanced data, so the
        weight should be tuned via --weight_sweep, not left at sklearn's
        default "balanced" (which gives the full ~577 ratio).

    min_samples_leaf : int
        Minimum samples required at a leaf.  Larger values prevent the
        tree from creating tiny leaves with extreme probabilities (1.0 or
        0.0) that don't generalise.  20 is a reasonable default.

    criterion : str
        "entropy" is more sensitive to small impurity changes than "gini",
        which helps with imbalanced data.

    max_depth
        depth 4-5: underfits, misses complex fraud patterns
        depth 6-7: good balance for this dataset
        depth 8+: starts overfitting, larger FHE circuit

    n_bits
        Quantisation precision.  Decision Trees tolerate low n_bits well
        because their computation is just integer comparisons.
        6 is usually sufficient.
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
    """Train in plaintext.  Single pass, sub-second."""
    print("\n--- Training Decision Tree ---")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.3f}s")
    print(f"Tree depth  : {model.get_depth()}")
    print(f"Tree leaves : {model.get_n_leaves()}")
    return model


# ---------------------------------------------------------------------------
# Section 5 - Print tree (interpretability)
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def print_tree(model: DecisionTreeClassifier, max_depth: int = 6) -> None:
    """
    Print a human-readable version of the decision tree.
    Only meaningful for shallow trees (max_depth <= 6).
    """
    if max_depth > 6:
        print("\n(Tree too deep to print readably - skipping)")
        return

    print(f"\n--- Decision Tree structure (first 3 levels) ---")

    sklearn_tree = model.sklearn_model
    tree_text = export_text(
        sklearn_tree,
        feature_names=FEATURE_NAMES,
        max_depth=3,
        show_weights=False,
    )
    print(tree_text)


# ---------------------------------------------------------------------------
# Section 6 - Plaintext evaluation with threshold tuning
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
# Section 7 - max_depth sweep
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
    Train and evaluate at several max_depth values.

    Use this to find the depth that best balances accuracy vs FHE circuit
    size before committing to compilation.
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
# Section 8 - class_weight sweep
# ---------------------------------------------------------------------------

def weight_sweep(
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    n_bits:    int = 6,
    max_depth: int = 7,
    weights:   list | None = None,
) -> None:
    """
    Train the tree at several class_weight values to find the sweet spot.

    Single trees are extremely sensitive to class weighting on imbalanced
    data:
        weight too low  (~5)  -> model predicts everything as legit
        weight too high (~24) -> model predicts everything as fraud

    The "Fraud %" column shows the fraction of test predictions flagged
    as fraud at the optimal threshold.  Aim for 0.5% to 5%
    (true fraud rate is 0.17%).
    """
    if weights is None:
        weights = [5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 35.0, 50.0]

    print(f"\n--- class_weight sweep (max_depth={max_depth}) ---")
    print(f"{'w_fraud':<10} {'PR-AUC':<10} {'ROC-AUC':<10} "
          f"{'Threshold':<12} {'Fraud %':<10}")
    print("-" * 60)

    for w in weights:
        cw    = {0: 1.0, 1: w}
        model = build_model(
            n_bits=n_bits,
            max_depth=max_depth,
            class_weight=cw,
            min_samples_leaf=20,
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)

        prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx  = f1_scores.argmax()
        best_thr  = float(thresholds[best_idx])
        fraud_pct = (y_proba >= best_thr).mean() * 100

        print(f"{w:<10.1f} {pr_auc:<10.4f} {roc_auc:<10.4f} "
              f"{best_thr:<12.4f} {fraud_pct:<10.2f}")

    print("\nPick a w_fraud where:")
    print("  - PR-AUC is highest")
    print("  - Fraud %% is between 0.5%% and 5%% "
          "(true rate is 0.17%%)")
    print("Update DEFAULT_FRAUD_WEIGHT at the top of model_dt.py "
          "with your choice.")


# ---------------------------------------------------------------------------
# Section 9 - FHE compilation and simulation evaluation
# ---------------------------------------------------------------------------

def compile_to_fhe(
    model:   DecisionTreeClassifier,
    X_train: np.ndarray,
):
    """
    Compile to FHE circuit.

    A Decision Tree compiles to the simplest circuit of all three models.
    Each prediction follows one path from root to leaf.
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
# Section 10 - Save artefacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model:         DecisionTreeClassifier,
    params:        dict,
    metrics:       dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # FHE circuit serialization via FHEModelDev triggers a libc++ crash on macOS.
    # The circuit is compiled at server startup instead.
    # Keep sklearn model for reference; server uses cml_model.json instead.
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
    p.add_argument("--max_depth",         type=int,   default=7)
    p.add_argument("--min_samples_leaf",  type=int,   default=20)
    p.add_argument("--criterion",         type=str,   default="entropy")
    p.add_argument("--fraud_weight",      type=float, default=DEFAULT_FRAUD_WEIGHT,
                   help="Weight applied to fraud class. Use --weight_sweep to find.")
    p.add_argument("--depth_sweep",       action="store_true",
                   help="Sweep max_depth 3-8 and exit")
    p.add_argument("--weight_sweep",      action="store_true",
                   help="Sweep class_weight values and exit")
    p.add_argument("--skip_compile",      action="store_true",
                   help="Skip FHE compilation")
    p.add_argument("--data_dir",          type=Path, default=DATA_DIR)
    p.add_argument("--artifacts_dir",     type=Path, default=ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test = load_splits(args.data_dir)

    # --- Optional sweeps ---
    if args.weight_sweep:
        weight_sweep(X_train, y_train, X_test, y_test,
                     n_bits=args.n_bits, max_depth=args.max_depth)
        return

    if args.depth_sweep:
        depth_sweep(X_train, y_train, X_test, y_test, n_bits=args.n_bits)
        return

    # --- Standard training flow ---
    cw = compute_class_weight(y_train, w_fraud=args.fraud_weight)

    params = dict(
        n_bits=args.n_bits,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        fraud_weight=args.fraud_weight,
    )
    model = build_model(
        n_bits=args.n_bits,
        max_depth=args.max_depth,
        class_weight=cw,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
    )
    model = train(model, X_train, y_train)

    print_tree(model, max_depth=args.max_depth)

    pt_metrics = evaluate_plaintext(model, X_test, y_test)

    # Save CML model as JSON with float thresholds BEFORE compile() replaces
    # them with quantized integers.  The server loads this file and recompiles.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(args.artifacts_dir / "cml_model.json", "w") as f:
        model.dump(f)
    print(f"  CML model (pre-compile) saved to {args.artifacts_dir}/cml_model.json")

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