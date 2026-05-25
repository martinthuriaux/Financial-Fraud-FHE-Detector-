"""
model_nn.py
-----------
Defines and trains a small quantisation-aware MLP for FHE fraud detection
using Brevitas (QAT layers), skorch (sklearn-compatible training loop), and
Concrete-ML (FHE compilation).

Dependency: preprocessing.py must have run first to produce the .npy splits.

Usage
-----
    python model_nn.py                        # default settings
    python model_nn.py --n_w_bits 6 --n_a_bits 6  # quantisation precision (max ~6 for FHE)
    python model_nn.py --smote               # enable SMOTE oversampling
    python model_nn.py --skip_compile        # train only, no FHE compile
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import brevitas.nn as qnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
from skorch.dataset import ValidSplit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from concrete.ml.sklearn import NeuralNetClassifier


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts" / "nn"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Section 1 — Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path) -> tuple[np.ndarray, ...]:
    """
    Load the processed arrays produced by preprocessing.py.

    dtype casting is important:
      - Features must be float32 (PyTorch default float type)
      - Labels must be int64 (what CrossEntropyLoss and skorch expect)
    """
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    X_test  = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    y_test  = np.load(data_dir / "y_test.npy").astype(np.int64)

    print(f"Train : {X_train.shape}  fraud rate: {(y_train == 1).mean():.3%}")
    print(f"Test  : {X_test.shape}   fraud rate: {(y_test == 1).mean():.3%}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Section 2 — SMOTE oversampling
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class (fraud) using SMOTE.

    SMOTE generates synthetic fraud examples by:
      1. Picking a real fraud example
      2. Finding its k nearest neighbours among other fraud examples
      3. Drawing a new point on the line between them

    This is more informative than simply duplicating existing rows because
    the model sees new, plausible points in feature space.

    Rules:
      - Apply to training set only (never test set)
      - Apply after the train/test split (already done in preprocessing.py)
      - Apply before building the model (class_weights depend on post-SMOTE counts)
    """
    print(f"\nBefore SMOTE — legit: {(y_train == 0).sum():,}  "
          f"fraud: {(y_train == 1).sum():,}")

    sm = SMOTE(random_state=random_state, sampling_strategy=0.1)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"After SMOTE  — legit: {(y_res == 0).sum():,}  "
          f"fraud: {(y_res == 1).sum():,}")

    return X_res.astype(np.float32), y_res.astype(np.int64)


# ---------------------------------------------------------------------------
# Section 3 — Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute per-class weights for CrossEntropyLoss.

    Even after SMOTE (which balances the dataset numerically), we keep a
    weighted loss as a second safeguard.  If SMOTE is skipped via
    Without --smote the weights become the primary mechanism for handling
    class imbalance.

    Weight formula:  w_i = total_samples / (n_classes * count_i)
    Concretely for binary:
        legit weight = 1.0
        fraud weight = n_legit / n_fraud

    CrossEntropyLoss multiplies each sample's loss by its class weight,
    so missing a fraud example costs proportionally more than a false alarm.
    """
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    w_fraud = n_legit / n_fraud
    weights = torch.tensor([1.0, w_fraud], dtype=torch.float32)
    print(f"\nClass weights — legit: 1.00,  fraud: {w_fraud:.2f}")
    return weights


# ---------------------------------------------------------------------------
# Section 4 — Build the NeuralNetClassifier
# ---------------------------------------------------------------------------

def build_model(
    n_w_bits:      int,
    n_a_bits:      int,
    class_weights: torch.Tensor | None = None,
    max_epochs:    int   = 100,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    patience:      int   = 10,
    prune_rate:    float = 0.1,
) -> NeuralNetClassifier:

    callbacks = [
        EpochScoring(
            scoring="average_precision",
            lower_is_better=False,
            on_train=False,
            name="valid_pr_auc",
        ),
        EarlyStopping(
            monitor="valid_pr_auc",
            patience=15,
            lower_is_better=False,
        ),
        LRScheduler(
            policy=CosineAnnealingLR,
            T_max=max_epochs,
            eta_min=1e-6,
        ),
    ]

    model = NeuralNetClassifier(
        # Shape of the built-in quantised MLP
        module__n_layers=2,
        module__n_hidden_neurons_multiplier=4,

        # Quantisation
        module__n_w_bits=n_w_bits,
        module__n_a_bits=n_a_bits,

        # Activation
        module__activation_function=nn.ReLU,
        module__n_prune_neurons_percentage=prune_rate,

        # Loss
        criterion=nn.CrossEntropyLoss,
        **({"criterion__weight": class_weights} if class_weights is not None else {}),

        # Optimiser
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,

        # Training
        max_epochs=max_epochs,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        train_split=ValidSplit(cv=0.1, stratified=True),
        callbacks=callbacks,
        verbose=1,
    )
    return model

# ---------------------------------------------------------------------------
# Section 5 — Evaluation
# ---------------------------------------------------------------------------

def evaluate_plaintext(model, X_test, y_test) -> dict:
    from sklearn.metrics import precision_recall_curve

    print("\n--- Plaintext evaluation ---")
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- find optimal threshold ---
    # precision_recall_curve returns precision, recall, and the threshold
    # that produced each pair. We pick the threshold with the best F1.
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx       = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    print(f"Default threshold : 0.5")
    print(f"Optimal threshold : {best_threshold:.4f}  "
          f"(precision={prec[best_idx]:.2f}, recall={rec[best_idx]:.2f})")

    # apply optimal threshold instead of default 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    print("\n-- With default threshold (0.50) --")
    print(classification_report(y_test, y_pred_default,
                                 target_names=["legit", "fraud"]))

    print(f"-- With optimal threshold ({best_threshold:.4f}) --")
    print(classification_report(y_test, y_pred_optimal,
                                 target_names=["legit", "fraud"]))

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}  ← primary metric under class imbalance")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_threshold,
    }

def evaluate_fhe(
    model: NeuralNetClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 50,
) -> dict:
    """
    Evaluate under FHE simulation.

    fhe="simulate" runs the quantised model without actual encryption.
    Results should be bit-identical to plaintext predictions.
    If they differ, it indicates a quantisation precision problem
    (try increasing n_bits).

    We limit to n_samples because real FHE execution (fhe="execute") is
    very slow — each forward pass can take several seconds.
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

    print(classification_report(y_sub, y_pred, target_names=["legit", "fraud"]))
    print(f"ROC-AUC (FHE sim) : {roc_auc:.4f}")
    print(f"Latency           : {per_sample * 1000:.1f} ms / sample")
    return {"roc_auc_fhe": roc_auc, "latency_ms": per_sample * 1000}


# ---------------------------------------------------------------------------
# Section 6 — FHE compilation and artefact saving
# ---------------------------------------------------------------------------

def compile_to_fhe(model: NeuralNetClassifier, X_train: np.ndarray):
    """
    Compile the trained model to an FHE circuit.

    Concrete-ML traces the quantised forward pass on representative data
    (X_train) to determine value ranges at every node, then lowers the
    computation to a Concrete circuit of boolean / integer operations.
    """
    print("\n--- Compiling to FHE circuit ---")
    t0 = time.perf_counter()
    fhe_circuit = model.compile(X_train)
    print(f"Compilation complete in {time.perf_counter() - t0:.1f}s")
    return fhe_circuit


def save_artifacts(
    model:        NeuralNetClassifier,
    params:       dict,
    metrics:      dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # FHE circuit serialization via FHEModelDev triggers a libc++ crash on macOS
    # (concrete-python LLVM bug). The circuit is compiled at server startup instead
    # — compile once, serve forever — so no disk serialization is needed here.

    # Use CML's own JSON serializer instead of joblib/pickle.
    # The model's PyTorch/Brevitas layers are not picklable (dynamically-injected
    # Brevitas classes have no stable module path). CML's dump() avoids this by
    # serializing weights via torch.save to a BytesIO buffer (hex-encoded in JSON)
    # and handles quantizers/ONNX via a custom ConcreteEncoder.
    # server.py recompiles the FHE circuit at startup, so we clear it first.
    # CML's serializer refuses to dump trained callback objects (EarlyStopping,
    # LRScheduler, etc.) and the compiled FHE circuit (ctypes pointers). Both are
    # irrelevant after training: callbacks are training-only, and server.py
    # recompiles the circuit at startup.
    qm = getattr(model, 'quantized_module_', None)
    if qm is not None and hasattr(qm, 'fhe_circuit'):
        qm.fhe_circuit = None

    # CML's deserializer reconstructs CrossEntropyLoss() with no arguments, so
    # it has no 'weight' buffer.  Loading a state dict that contains 'weight'
    # into that bare criterion raises RuntimeError.  Strip it here — the class
    # weight is only used during training, not inference.
    crit = getattr(getattr(model, 'sklearn_model', None), 'criterion_', None)
    if crit is not None and 'weight' in crit._buffers:
        del crit._buffers['weight']

    model.callbacks = "disable"
    with open(artifacts_dir / "cml_model.json", "w") as f:
        model.dump(f)

    with open(artifacts_dir / "threshold.json", "w") as f:
        json.dump({"threshold": float(metrics.get("best_threshold", 0.5))}, f)

    with open(artifacts_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    metrics_to_save = {k: float(v) for k, v in metrics.items()
                       if k != "best_threshold"}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nArtefacts saved to {artifacts_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compile the FHE MLP fraud detector")
    p.add_argument("--n_w_bits",       type=int,   default=6)
    p.add_argument("--n_a_bits",       type=int,   default=6)
    p.add_argument("--max_epochs",    type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--prune_rate",     type=float, default=0.0)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--smote",         action="store_true")
    p.add_argument("--skip_compile",  action="store_true")
    p.add_argument("--data_dir",      type=Path,  default=DATA_DIR)
    p.add_argument("--artifacts_dir", type=Path,  default=ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load data
    X_train, X_test, y_train, y_test = load_splits(args.data_dir)

    # 2. Oversample minority class in training set (opt-in)
    if args.smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # 3. Class weights from post-SMOTE labels — always applied.
    # With full SMOTE (1:1) w_fraud becomes 1.0 (no effect).
    # With partial SMOTE (e.g. 0.1) or no SMOTE, w_fraud reflects the residual imbalance.
    class_weights = compute_class_weights(y_train)

    # 4. Build model
    params = dict(
        n_w_bits=args.n_w_bits,
        n_a_bits=args.n_a_bits,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        prune_rate=args.prune_rate,
    )

    model = build_model(
        class_weights=class_weights,
        **params,        )

    # 5. Train
    print("\n--- Training ---")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Training complete in {time.perf_counter() - t0:.1f}s")

    # 6. Plaintext evaluation (always run, fast)
    pt_metrics = evaluate_plaintext(model, X_test, y_test)

    if args.skip_compile:
        print("\nSkipping FHE compilation (--skip_compile set).")
        return

    # 7. Compile → evaluate under FHE simulation → save
    compile_to_fhe(model, X_train)
    fhe_metrics = evaluate_fhe(model, X_test, y_test)

    delta = abs(pt_metrics["roc_auc"] - fhe_metrics["roc_auc_fhe"])
    print(f"\nROC-AUC delta (plaintext vs FHE sim): {delta:.4f}")
    if delta > 0.02:
        print("  Warning: delta > 0.02 — consider increasing n_bits.")

    save_artifacts(model, params, {**pt_metrics, **fhe_metrics},
                   args.artifacts_dir)


if __name__ == "__main__":
    main()