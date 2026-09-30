"""
Parking Slot Occupancy Prediction Module
----------------------------------------
Loads the trained LSTM model and predicts the future occupancy probability
for a specific parking slot (P1, P2, P3, P4) based on its latest observation sequence.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Paths
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "parking_lstm.keras")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data.csv")

# Global cached model instance to avoid reloading overhead
_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads and caches the trained Keras LSTM model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run 'dnn/train_lstm.py' first."
        )

    import tensorflow as tf
    _CACHED_MODEL = tf.keras.models.load_model(model_path)
    return _CACHED_MODEL


def get_latest_sequence_for_slot(
    slot_id: str,
    csv_path: str = DEFAULT_CSV_PATH,
    window_size: int = 8
) -> np.ndarray:
    """
    Extracts the most recent `window_size` occupancy observations for a given slot.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    slot_df = df[df["slot_id"] == slot_id].sort_values(by="timestamp").reset_index(drop=True)
    if len(slot_df) < window_size:
        raise ValueError(
            f"Slot '{slot_id}' has only {len(slot_df)} records, but window_size is {window_size}."
        )

    recent_occupancy = slot_df["occupied"].tail(window_size).values.astype(np.float32)
    # Reshape to (1, window_size, 1) for LSTM input
    return recent_occupancy.reshape(1, window_size, 1)


def predict_slot_occupancy(
    slot_id: str,
    model_path: str = DEFAULT_MODEL_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    window_size: int = 8,
    threshold: float = 0.5,
    custom_sequence: Optional[list] = None
) -> Dict[str, Any]:
    """
    Predicts whether a slot will be OCCUPIED or AVAILABLE in the next interval.

    Returns:
        dict: {
            "slot_id": str,
            "probability": float,
            "prediction": "OCCUPIED" | "AVAILABLE",
            "threshold": float,
            "sequence_used": list
        }
    """
    slot_id = slot_id.strip().upper()
    valid_slots = ["P1", "P2", "P3", "P4"]
    if slot_id not in valid_slots:
        raise ValueError(f"Invalid slot '{slot_id}'. Allowed slots: {', '.join(valid_slots)}")

    model = load_model(model_path)

    if custom_sequence is not None:
        if len(custom_sequence) != window_size:
            raise ValueError(f"custom_sequence must have length {window_size}, got {len(custom_sequence)}")
        input_seq = np.array(custom_sequence, dtype=np.float32).reshape(1, window_size, 1)
        used_seq = list(custom_sequence)
    else:
        input_seq = get_latest_sequence_for_slot(slot_id, csv_path=csv_path, window_size=window_size)
        used_seq = input_seq.flatten().tolist()

    # Predict occupancy probability (between 0.0 and 1.0)
    raw_pred = model.predict(input_seq, verbose=0)
    probability = float(raw_pred[0][0])
    prediction = "OCCUPIED" if probability >= threshold else "AVAILABLE"

    return {
        "slot_id": slot_id,
        "probability": round(probability, 4),
        "prediction": prediction,
        "threshold": threshold,
        "sequence_used": [int(x) for x in used_seq]
    }


def predict_all_slots(
    model_path: str = DEFAULT_MODEL_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    window_size: int = 8,
    threshold: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """Predicts future occupancy for all 4 slots (P1 to P4)."""
    results = {}
    for slot in ["P1", "P2", "P3", "P4"]:
        results[slot] = predict_slot_occupancy(
            slot_id=slot,
            model_path=model_path,
            csv_path=csv_path,
            window_size=window_size,
            threshold=threshold
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict parking slot occupancy using trained LSTM")
    parser.add_argument("--slot", type=str, default=None, help="Target slot ID (P1, P2, P3, P4). If omitted, predicts for all.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to parking CSV")
    parser.add_argument("--window", type=int, default=8, help="Sequence window size (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("      LSTM PARKING OCCUPANCY PREDICTION")
    print("=" * 50)

    if args.slot:
        res = predict_slot_occupancy(
            slot_id=args.slot,
            model_path=args.model,
            csv_path=args.csv,
            window_size=args.window,
            threshold=args.threshold
        )
        print(f"Slot:                             {res['slot_id']}")
        print(f"Recent History Sequence:          {res['sequence_used']}")
        print(f"Predicted Occupancy Probability:  {res['probability']:.4f}")
        print(f"Prediction:                       {res['prediction']}")
    else:
        all_results = predict_all_slots(
            model_path=args.model,
            csv_path=args.csv,
            window_size=args.window,
            threshold=args.threshold
        )
        for slot, res in all_results.items():
            print(f"Slot {slot}: Probability = {res['probability']:.4f} -> {res['prediction']:<9} (Sequence: {res['sequence_used']})")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
