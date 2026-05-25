"""
client.py
---------
Test client for the FHE fraud detection server.

Picks transactions from the saved test set, sends them to the server via
HTTP, prints predictions and timing.

Three modes:
    single  : send one transaction (default — a known fraud case)
    batch   : send N transactions, summarise accuracy + latency
    compare : send the same transaction with fhe=true and fhe=false,
              print the latency difference

Usage
-----
    python client.py                        # single fraud case, FHE enabled
    python client.py --mode batch --n 20    # 20 transactions
    python client.py --mode compare         # FHE vs plaintext on one sample
    python client.py --no_fhe               # disable FHE for speed testing
    python client.py --host 192.168.1.5     # remote server
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "splits"


# ---------------------------------------------------------------------------
# Section 1 — Server communication
# ---------------------------------------------------------------------------

def call_predict(
    server_url: str,
    features:   np.ndarray,
    fhe:        bool = True,
    scaled:     bool = True,
) -> dict:
    """
    Send a single transaction to the /predict endpoint.

    Parameters
    ----------
    server_url : str
        Base URL of the server, e.g. "http://localhost:8000"
    features : np.ndarray
        1D array of 30 floats — one transaction.
    fhe : bool
        If True, server runs the FHE circuit (slow but private).
        If False, server runs the model in plaintext (fast).
    scaled : bool
        If True, tells the server the features are already scaled
        and to skip the StandardScaler step.

    Returns
    -------
    dict : the server's response, with keys
        prediction, probability, threshold, latency_ms, model, fhe_executed
    """
    url = f"{server_url}/predict"

    # Build the request payload.  Pydantic on the server side will reject
    # any malformed input automatically.
    payload = {"features": features.tolist()}

    # Query parameters control server behaviour without changing the body
    params  = {"fhe": str(fhe).lower(), "scaled": str(scaled).lower()}

    response = requests.post(url, json=payload, params=params)

    # Raise an exception if the server returned an error (4xx or 5xx)
    response.raise_for_status()

    return response.json()


def check_health(server_url: str) -> None:
    """
    Confirm the server is alive before sending anything else.
    Exits the script with a helpful message if the server is unreachable.
    """
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"Server is healthy. Serving model: {info['model']}\n")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: cannot reach server at {server_url}")
        print("Did you start it with `MODEL=xgboost python server.py`?")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Section 2 — Data loading
# ---------------------------------------------------------------------------

def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the held-out test set saved by preprocessing.py.
    These transactions are already scaled, so we set scaled=True
    when sending them to the server.
    """
    X_test = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    return X_test, y_test


def pick_one_fraud(X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pick a deterministic fraud case from the test set.
    Returns (features, true_label).
    """
    fraud_idx = np.where(y_test == 1)[0]
    chosen    = fraud_idx[0]   # always the first fraud — reproducible
    return X_test[chosen], int(y_test[chosen])


# ---------------------------------------------------------------------------
# Section 3 — Result formatting
# ---------------------------------------------------------------------------

def print_prediction(result: dict, true_label: int | None = None) -> None:
    """Print a single /predict response in a readable format."""
    pred = result["prediction"]
    prob = result["probability"]
    thr  = result["threshold"]
    lat  = result["latency_ms"]
    fhe  = result["fhe_executed"]

    mode = "FHE" if fhe else "plaintext"

    print(f"  Mode        : {mode}")
    print(f"  Probability : {prob:.4f}")
    print(f"  Threshold   : {thr:.4f}")
    print(f"  Prediction  : {pred}")
    print(f"  Latency     : {lat:.1f} ms")

    if true_label is not None:
        true_str = "FRAUD" if true_label == 1 else "LEGIT"
        match    = "correct" if pred == true_str else "wrong"
        print(f"  Truth       : {true_str}  ({match})")


# ---------------------------------------------------------------------------
# Section 4 — Modes
# ---------------------------------------------------------------------------

def run_single(server_url: str, use_fhe: bool) -> None:
    """Send a single known fraud case and print the result."""
    print("=" * 50)
    print(f"SINGLE TRANSACTION — fhe={use_fhe}")
    print("=" * 50)

    X_test, y_test = load_test_set()
    features, true_label = pick_one_fraud(X_test, y_test)

    print(f"Sending transaction (true label: FRAUD)...")
    result = call_predict(server_url, features, fhe=use_fhe)
    print_prediction(result, true_label=true_label)


def run_compare(server_url: str) -> None:
    """Send the same transaction twice — once with FHE, once without."""
    print("=" * 50)
    print("COMPARISON — FHE vs plaintext on the same transaction")
    print("=" * 50)

    X_test, y_test = load_test_set()
    features, true_label = pick_one_fraud(X_test, y_test)

    print("\n--- Plaintext inference ---")
    r_plain = call_predict(server_url, features, fhe=False)
    print_prediction(r_plain, true_label=true_label)

    print("\n--- FHE inference ---")
    r_fhe = call_predict(server_url, features, fhe=True)
    print_prediction(r_fhe, true_label=true_label)

    # Compare
    speedup = r_fhe["latency_ms"] / r_plain["latency_ms"]
    print(f"\nFHE is {speedup:.1f}x slower than plaintext")

    # Both should produce the same prediction.  If they don't, n_bits is
    # probably too low and quantisation is causing different outputs.
    if r_plain["prediction"] != r_fhe["prediction"]:
        print("WARNING: predictions disagree — possible quantisation issue")


def run_batch(server_url: str, n: int, use_fhe: bool) -> None:
    """
    Send N random transactions, half fraud and half legit if possible.
    Print individual results and an accuracy summary.
    """
    print("=" * 50)
    print(f"BATCH — {n} transactions, fhe={use_fhe}")
    print("=" * 50)

    X_test, y_test = load_test_set()

    # Stratified sample — half fraud, half legit if possible
    rng       = np.random.default_rng(42)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_fraud   = min(len(fraud_idx), n // 2)
    n_legit   = n - n_fraud
    indices   = np.concatenate([
        rng.choice(fraud_idx, n_fraud, replace=False),
        rng.choice(legit_idx, n_legit, replace=False),
    ])
    rng.shuffle(indices)

    # Send each transaction and tally results
    n_correct        = 0
    n_false_positive = 0   # flagged as fraud but actually legit
    n_false_negative = 0   # flagged as legit but actually fraud
    total_latency    = 0.0

    print(f"\nSending {n} transactions...\n")

    for i, idx in enumerate(indices, start=1):
        result = call_predict(server_url, X_test[idx], fhe=use_fhe)
        truth  = "FRAUD" if y_test[idx] == 1 else "LEGIT"
        pred   = result["prediction"]

        total_latency += result["latency_ms"]

        if pred == truth:
            n_correct += 1
            status = "✓"
        elif pred == "FRAUD":
            n_false_positive += 1
            status = "✗ (false alarm)"
        else:
            n_false_negative += 1
            status = "✗ (missed fraud)"

        print(f"  [{i:3d}/{n}]  pred={pred:<5s}  true={truth:<5s}  "
              f"prob={result['probability']:.3f}  {status}")

    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY")
    print("=" * 50)
    print(f"Accuracy           : {n_correct}/{n}  ({n_correct/n:.1%})")
    print(f"False positives    : {n_false_positive}  "
          f"(innocent transactions flagged)")
    print(f"False negatives    : {n_false_negative}  "
          f"(fraud transactions missed)")
    print(f"Average latency    : {total_latency/n:.1f} ms")
    print(f"Total time         : {total_latency/1000:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FHE fraud detection client")
    p.add_argument("--mode",   choices=["single", "batch", "compare"],
                   default="single")
    p.add_argument("--n",      type=int, default=10,
                   help="Number of transactions for batch mode")
    p.add_argument("--no_fhe", action="store_true",
                   help="Disable FHE for faster testing")
    p.add_argument("--host",   default="localhost")
    p.add_argument("--port",   type=int, default=8000)
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    server_url = f"http://{args.host}:{args.port}"
    use_fhe    = not args.no_fhe

    check_health(server_url)

    if args.mode == "single":
        run_single(server_url, use_fhe)
    elif args.mode == "batch":
        run_batch(server_url, args.n, use_fhe)
    elif args.mode == "compare":
        run_compare(server_url)


if __name__ == "__main__":
    main()