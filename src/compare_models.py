"""
compare_models.py
-----------------
Benchmark the currently-running server and accumulate results across
all three models for the final comparison table.

Workflow
--------
Only one server can run at a time (port 8000 is shared).  This script
queries whichever server is running, saves the result to results.json,
and the --report flag prints the accumulated table.

    # Terminal 1                          # Terminal 2
    MODEL=decision_tree python server.py  python compare_models.py
    [Ctrl+C the server]
    MODEL=xgboost python server.py        python compare_models.py
    [Ctrl+C]
    MODEL=nn python server.py             python compare_models.py
    [Ctrl+C]
                                          python compare_models.py --report

The final --report call prints the comparison table.  Each measurement
uses the SAME test transactions across all three models so the numbers
are directly comparable.

Usage
-----
    python compare_models.py                  # benchmark running server, n=20
    python compare_models.py --n 30          # use 30 test samples
    python compare_models.py --report        # print accumulated table
    python compare_models.py --reset         # clear results.json and exit
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data" / "splits"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_FILE  = ROOT / "results.json"


# ---------------------------------------------------------------------------
# Section 1 — Pick the same test transactions every run
# ---------------------------------------------------------------------------

def get_test_indices(n: int, seed: int = 42) -> np.ndarray:
    """
    Pick the same N test set indices on every invocation.

    Uses a fixed random seed so all three models are evaluated on
    exactly the same transactions — making FHE latency and accuracy
    numbers directly comparable.

    Stratified: roughly half fraud, half legit.
    """
    y_test = np.load(DATA_DIR / "y_test.npy").astype(np.int64)

    rng       = np.random.default_rng(seed)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_fraud   = min(len(fraud_idx), n // 2)
    n_legit   = n - n_fraud

    indices = np.concatenate([
        rng.choice(fraud_idx, n_fraud, replace=False),
        rng.choice(legit_idx, n_legit, replace=False),
    ])
    rng.shuffle(indices)
    return indices


# ---------------------------------------------------------------------------
# Section 2 — Server queries
# ---------------------------------------------------------------------------

def check_server(server_url: str) -> str:
    """
    Confirm a server is running and return the name of the model it serves.
    Raises SystemExit with a helpful message if unreachable.
    """
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()["model"]
    except requests.exceptions.ConnectionError:
        print(f"ERROR: cannot reach server at {server_url}")
        print("Start one with: MODEL=<decision_tree|xgboost|nn> python server.py")
        raise SystemExit(1)


def call_predict(
    server_url: str,
    features:   np.ndarray,
    fhe:        bool,
) -> dict:
    """
    Single /predict call.  Uses a generous 120s timeout because XGBoost
    under FHE can take 5–10s per prediction.
    """
    payload  = {"features": features.tolist()}
    params   = {"fhe": str(fhe).lower(), "scaled": "true"}
    response = requests.post(
        f"{server_url}/predict",
        json=payload,
        params=params,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Section 3 — Benchmark one model
# ---------------------------------------------------------------------------

def benchmark_model(
    server_url: str,
    model_name: str,
    n_samples:  int,
) -> dict:
    """
    Run N transactions twice — once with fhe=True, once with fhe=False.
    Record latencies and predictions.  Compute accuracy and median timings.

    Two passes is intentional: same inputs, different execution modes,
    so we can directly compare them on identical data.
    """
    # Static info from disk — these are training-time metrics
    artifacts_dir = ARTIFACTS_DIR / model_name
    with open(artifacts_dir / "metrics.json") as f:
        train_metrics = json.load(f)
    with open(artifacts_dir / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    # Load test set features (already scaled)
    X_test  = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test  = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    indices = get_test_indices(n_samples)

    print(f"\nBenchmarking {model_name} on {n_samples} transactions...")
    print(f"(estimated total time: ")
    if model_name == "decision_tree":
        print(f"  ~{n_samples * 0.5:.0f}s for FHE + ~1s plaintext)")
    elif model_name == "xgboost":
        print(f"  ~{n_samples * 6:.0f}s for FHE + ~1s plaintext)")
    else:
        print(f"  ~{n_samples * 10:.0f}s for FHE + ~2s plaintext)")

    # --- Plaintext pass (fast) ---
    print(f"\n  Plaintext pass...")
    plaintext_latencies = []
    plaintext_preds     = []
    t_start             = time.perf_counter()
    for idx in indices:
        result = call_predict(server_url, X_test[idx], fhe=False)
        plaintext_latencies.append(result["latency_ms"])
        plaintext_preds.append(result["prediction"])
    plaintext_total = time.perf_counter() - t_start
    print(f"    Done in {plaintext_total:.1f}s "
          f"(median {statistics.median(plaintext_latencies):.1f} ms/sample)")

    # --- FHE pass (slow) ---
    print(f"  FHE pass...")
    fhe_latencies = []
    fhe_preds     = []
    t_start       = time.perf_counter()
    for i, idx in enumerate(indices, start=1):
        result = call_predict(server_url, X_test[idx], fhe=True)
        fhe_latencies.append(result["latency_ms"])
        fhe_preds.append(result["prediction"])
        print(f"    [{i}/{n_samples}] "
              f"{result['latency_ms']:.0f}ms  "
              f"pred={result['prediction']}")
    fhe_total = time.perf_counter() - t_start
    print(f"    Done in {fhe_total:.1f}s "
          f"(median {statistics.median(fhe_latencies):.1f} ms/sample)")

    # --- Accuracy on this sample ---
    true_labels = ["FRAUD" if y_test[i] == 1 else "LEGIT" for i in indices]
    n_correct   = sum(1 for p, t in zip(fhe_preds, true_labels) if p == t)
    accuracy    = n_correct / n_samples

    # Check that plaintext and FHE agree (sanity check)
    agreement = sum(1 for p, f in zip(plaintext_preds, fhe_preds) if p == f)
    if agreement < n_samples:
        print(f"  WARNING: plaintext and FHE disagreed on "
              f"{n_samples - agreement} samples — possible quantisation loss")

    # Median is more robust than mean to outliers (first call is always
    # slower due to circuit warmup)
    return {
        "model":              model_name,
        "n_samples":          n_samples,
        "pr_auc_train":       train_metrics.get("pr_auc"),
        "roc_auc_train":      train_metrics.get("roc_auc"),
        "threshold":          threshold,
        "accuracy_fhe":       accuracy,
        "plaintext_ms":       statistics.median(plaintext_latencies),
        "fhe_ms":             statistics.median(fhe_latencies),
        "slowdown_factor":    (statistics.median(fhe_latencies)
                               / max(statistics.median(plaintext_latencies),
                                     0.001)),
        "plaintext_total_s":  plaintext_total,
        "fhe_total_s":        fhe_total,
    }


# ---------------------------------------------------------------------------
# Section 4 — Persist results across runs
# ---------------------------------------------------------------------------

def load_results() -> dict:
    """Load accumulated results, or empty dict if none exist yet."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_result(result: dict) -> None:
    """Add this model's result to the accumulated results file."""
    results = load_results()
    results[result["model"]] = result
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResult saved to {RESULTS_FILE}")
    print(f"Models benchmarked so far: {list(results.keys())}")


# ---------------------------------------------------------------------------
# Section 5 — Print the comparison report
# ---------------------------------------------------------------------------

def print_report() -> None:
    """Print the final comparison table from accumulated results."""
    results = load_results()
    if not results:
        print(f"No results found at {RESULTS_FILE}")
        print("Run the benchmark with each model first.")
        return

    # Display models in a logical order if all are present
    order = ["decision_tree", "xgboost", "nn"]
    ordered = [results[m] for m in order if m in results]

    print("\n" + "=" * 90)
    print("FHE FRAUD DETECTION — MODEL COMPARISON")
    print("=" * 90)

    # Header
    header = (f"{'Model':<16} {'PR-AUC':<8} {'ROC-AUC':<9} "
              f"{'Accuracy':<10} {'Plaintext':<11} {'FHE':<13} "
              f"{'Slowdown':<10}")
    print(header)
    print("-" * 90)

    for r in ordered:
        print(
            f"{r['model']:<16} "
            f"{r['pr_auc_train']:<8.4f} "
            f"{r['roc_auc_train']:<9.4f} "
            f"{r['accuracy_fhe']:<10.1%} "
            f"{r['plaintext_ms']:<11.1f} "
            f"{r['fhe_ms']:<13.1f} "
            f"{r['slowdown_factor']:<10.0f}×"
        )

    print("-" * 90)
    print(f"  PR-AUC, ROC-AUC : threshold-independent training metrics")
    print(f"  Accuracy        : on {ordered[0]['n_samples']} held-out test "
          f"transactions under FHE")
    print(f"  Plaintext / FHE : median ms per single-sample prediction")
    print(f"  Slowdown        : FHE latency / plaintext latency")
    print("=" * 90)

    # Optional summary observations
    if len(ordered) >= 2:
        print("\nKey findings:")
        best_acc = max(ordered, key=lambda r: r["pr_auc_train"])
        fastest  = min(ordered, key=lambda r: r["fhe_ms"])
        print(f"  Most accurate    : {best_acc['model']}  "
              f"(PR-AUC {best_acc['pr_auc_train']:.4f})")
        print(f"  Fastest under FHE: {fastest['model']}  "
              f"({fastest['fhe_ms']:.0f} ms / sample)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-model FHE benchmark and comparison"
    )
    p.add_argument("--n",      type=int, default=20,
                   help="Number of test transactions to benchmark")
    p.add_argument("--host",   default="localhost")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--report", action="store_true",
                   help="Print accumulated results table and exit")
    p.add_argument("--reset",  action="store_true",
                   help="Delete results.json and exit")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    server_url = f"http://{args.host}:{args.port}"

    if args.reset:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
            print(f"Deleted {RESULTS_FILE}")
        else:
            print("No results file to delete.")
        return

    if args.report:
        print_report()
        return

    # Benchmark whichever model is currently being served
    model_name = check_server(server_url)
    print(f"Server is healthy. Currently serving: {model_name}")

    result = benchmark_model(server_url, model_name, args.n)
    save_result(result)

    # Print intermediate results so far
    print_report()


if __name__ == "__main__":
    main()