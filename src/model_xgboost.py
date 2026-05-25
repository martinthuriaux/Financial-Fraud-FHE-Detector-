"""
model_xgboost.py
----------------
Trains a quantisation-aware XGBoost classifier with Concrete-ML,
compiles it to an FHE circuit, and saves all artefacts.

Concrete-ML's XGBClassifier is a drop-in replacement for the standard
XGBoost sklearn API.  The only FHE-specific additions are:
  - n_bits       : quantisation precision for inputs and tree thresholds
  - .compile()   : lowers the trained model to an FHE circuit
  - fhe=         : argument on .predict() to select plaintext / simulate / execute

Dependency: preprocessing.py must have run first.

Usage
-----
    python model_xgboost.py                   # default settings
    python model_xgboost.py --sweep           # compare n_bits 4-8, no compile
    python model_xgboost.py --n_bits 8        # specific bit width
    python model_xgboost.py --skip_compile    # train + evaluate, no FHE compile
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
from concrete.ml.sklearn import XGBClassifier


# ---------------------------------------------------------------------------
# Section 1 - Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts" / "xgboost"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Section 2 - Load splits
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path) -> tuple[np.ndarray, ...]:
    """
    Load the preprocessed arrays produced by preprocessing.py.

    XGBoost accepts float32 natively.  Labels are kept as integers.
    """
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    X_test  = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    y_test  = np.load(data_dir / "y_test.npy").astype(np.int64)

    print(f"Train : {X_train.shape}  fraud rate: {(y_train == 1).mean():.3%}")
    print(f"Test  : {X_test.shape}   fraud rate: {(y_test == 1).mean():.3%}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Section 3 - Class imbalance via scale_pos_weight
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute the scale_pos_weight parameter for XGBoost.

    XGBoost multiplies the gradient of every positive (fraud) example by
    this value, making the model treat each fraud case as if it were
    scale_pos_weight fraud cases during training.

    Formula: sqrt(n_legit / n_fraud)

    The raw ratio (~577) is too aggressive: it pushes recall to near-100%
    but collapses precision because every gradient update for fraud is
    amplified 577×.  Taking the square root (~24) moderates the signal —
    fraud examples are still strongly up-weighted, but the model retains
    enough balance to avoid flagging everything as fraud.
    """
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    spw     = (n_legit / n_fraud) ** 0.5
    print(f"\nscale_pos_weight : {spw:.2f}  "
          f"({n_legit:,} legit / {n_fraud:,} fraud)")
    return float(spw)


# ---------------------------------------------------------------------------
# Section 4 - Build and train
# ---------------------------------------------------------------------------

def build_model(
    n_bits:            int,
    scale_pos_weight:  float,
    n_estimators:      int   = 200,
    max_depth:         int   = 5,
    learning_rate:     float = 0.05,
    subsample:         float = 0.8,
    colsample_bytree:  float = 0.8,
    min_child_weight:  int   = 5,
) -> XGBClassifier:
    """
    Construct the Concrete-ML XGBClassifier.

    Concrete-ML's XGBClassifier is API-compatible with the standard
    XGBoost sklearn wrapper.  The only FHE-specific parameter here is
    n_bits, which controls how inputs and tree split thresholds are
    quantised before compilation.

    Hyperparameter notes
    --------------------
    n_bits : int
        Lower = faster FHE circuit but potential accuracy loss.
        8 is a good starting point.  Use --sweep to find the best value
        before committing to a full FHE compile.

    n_estimators : int
        Number of trees.  More trees = more circuit operations = slower FHE.
        200 with early stopping typically converges well before the cap.

    max_depth : int
        Depth of each tree.  Deeper trees capture more complex interactions
        but produce larger circuits.  Keep at 4-6 for FHE.

    learning_rate : float
        Step size for each tree's contribution.  Lower learning rate with
        more estimators generally outperforms higher learning rate with fewer,
        but increases training time.

    subsample / colsample_bytree : float
        Fraction of rows / features sampled per tree.  Adds regularisation
        by introducing randomness, reducing overfitting.
    """
    model = XGBClassifier(
        n_bits=n_bits,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
    )
    return model


def train(
    model:   XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """
    Train the model in plaintext on the full training set.

    Early stopping was removed because the fraud class is too small (~394
    cases) to produce a reliable PR-AUC on a 10% validation split (~39
    fraud examples).  The noise caused early stopping to fire at iteration 3,
    leaving the model with only 3 trees.  Training all n_estimators on the
    full data is more reliable and takes ~4 minutes, which is acceptable
    for a one-off training run.
    """
    print("\n--- Training XGBoost ---")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.2f}s")
    return model


# ---------------------------------------------------------------------------
# Section 5 - Plaintext evaluation with threshold tuning
# ---------------------------------------------------------------------------

def evaluate_plaintext(
    model:  XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate in plaintext and find the optimal decision threshold.

    PR-AUC and ROC-AUC are threshold-independent.
    The classification report depends on the chosen threshold.
    We report both 0.50 and the optimal threshold for comparison.
    """
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
# Section 6 - n_bits sensitivity sweep
# ---------------------------------------------------------------------------

def n_bits_sweep(
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    X_test:           np.ndarray,
    y_test:           np.ndarray,
    scale_pos_weight: float,
    bits_range:       list | None = None,
) -> None:
    """
    Train and evaluate at several n_bits values without FHE compilation.

    Use this to find the best quantisation precision before committing to
    the slow compilation step.

    Lower n_bits  faster FHE circuit, risk of accuracy loss.
    Higher n_bits  slower FHE circuit, closer to full-precision accuracy.
    """
    if bits_range is None:
        bits_range = [4, 5, 6, 7, 8]

    print("\n--- n_bits sensitivity sweep ---")
    print(f"{'n_bits':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'Train time'}")
    print("-" * 40)

    results = {}
    for n in bits_range:
        model = build_model(n_bits=n, scale_pos_weight=scale_pos_weight)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)

        print(f"{n:<8} {roc_auc:<10.4f} {pr_auc:<10.4f} {elapsed:.2f}s")
        results[n] = {"roc_auc": roc_auc, "pr_auc": pr_auc}

    best_pr     = max(v["pr_auc"] for v in results.values())
    threshold   = best_pr * 0.99
    recommended = min(
        n for n, v in results.items() if v["pr_auc"] >= threshold
    )
    print(f"\nRecommended n_bits: {recommended}  "
          f"(PR-AUC within 1% of best at {best_pr:.4f})")


# ---------------------------------------------------------------------------
# Section 7 - FHE compilation and simulation evaluation
# ---------------------------------------------------------------------------

def compile_to_fhe(
    model:   XGBClassifier,
    X_train: np.ndarray,
):
    """
    Compile the trained XGBoost model to an FHE circuit.

    Concrete-ML traces the quantised decision tree logic on the training
    data to determine value ranges at every node, then lowers the
    computation to a Concrete circuit of integer operations.
    """
    print("\n--- Compiling to FHE circuit ---")
    t0 = time.perf_counter()
    fhe_circuit = model.compile(X_train)
    elapsed = time.perf_counter() - t0
    print(f"Compilation complete in {elapsed:.1f}s")
    return fhe_circuit


def evaluate_fhe(
    model:     XGBClassifier,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    n_samples: int = 100,
) -> dict:
    """
    Evaluate under FHE simulation on a subset of the test set.

    fhe="simulate" runs quantised integer computation without encryption.
    Results should be near-identical to plaintext.
    Delta > 0.02 on ROC-AUC suggests quantisation loss - increase n_bits.
    """
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
    model:         XGBClassifier,
    params:        dict,
    metrics:       dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # FHE circuit serialization via FHEModelDev triggers a libc++ crash on macOS
    # (concrete-python LLVM bug). The circuit is compiled at server startup instead
    # — compile once, serve forever — so no disk serialization is needed here.

    # Keep sklearn model for reference/diagnostics.
    # NOTE: by this point compile() has already quantized the thresholds to integers.
    # The server uses cml_model.json (saved before compile) for correct inference.
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
        description="Train and compile XGBoost FHE fraud detector"
    )
    p.add_argument("--n_bits",                type=int,   default=10)
    p.add_argument("--n_estimators",          type=int,   default=200)
    p.add_argument("--max_depth",             type=int,   default=5)
    p.add_argument("--learning_rate",         type=float, default=0.05)
    p.add_argument("--subsample",             type=float, default=0.8)
    p.add_argument("--colsample_bytree",      type=float, default=0.8)
    p.add_argument("--min_child_weight",      type=int,   default=5)
    p.add_argument("--sweep",            action="store_true",
                   help="Run n_bits sensitivity sweep and exit")
    p.add_argument("--skip_compile",     action="store_true",
                   help="Skip FHE compilation")
    p.add_argument("--data_dir",         type=Path,  default=DATA_DIR)
    p.add_argument("--artifacts_dir",    type=Path,  default=ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test = load_splits(args.data_dir)
    spw = compute_scale_pos_weight(y_train)

    if args.sweep:
        n_bits_sweep(X_train, y_train, X_test, y_test, spw)
        return

    params = dict(
        n_bits=args.n_bits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
    )
    model = build_model(scale_pos_weight=spw, **params)
    model = train(model, X_train, y_train)

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