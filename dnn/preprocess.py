"""
Data Preprocessing Module for LSTM Occupancy Prediction
-------------------------------------------------------
Reads raw historical sensor readings from 'data/parking_data.csv',
extracts time features, sorts chronologically by slot, builds sliding
window sequences, and splits into train/test sets without data leakage.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Path defaults
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data.csv")
DEFAULT_PROCESSED_NPZ = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.npz")
DEFAULT_PROCESSED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.csv")

SLOTS = ["P1", "P2", "P3", "P4"]


def load_raw_data(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Loads raw CSV data and converts timestamps to datetime objects.
    Ensures strict chronological sorting per slot to prevent data leakage.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["minute"] = df["timestamp"].dt.minute

    # Strictly sort by slot_id and timestamp
    df = df.sort_values(by=["slot_id", "timestamp"]).reset_index(drop=True)
    return df


def create_sliding_window_sequences(
    df: pd.DataFrame,
    window_size: int = 8,
    train_ratio: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Creates chronological sliding window sequences for each slot independently.
    
    For each slot:
      - Observations are split chronologically (first train_ratio into train, rest into test).
      - Sequences: Input X = [occ_t-8, ..., occ_t-1], Target y = occ_t.
      - Splitting BEFORE sequence generation across the boundary prevents boundary leakage.
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for slot in sorted(df["slot_id"].unique()):
        slot_df = df[df["slot_id"] == slot].sort_values(by="timestamp").reset_index(drop=True)
        occupancy_values = slot_df["occupied"].values.astype(np.float32)

        total_obs = len(occupancy_values)
        if total_obs <= window_size:
            print(f"[WARNING] Not enough observations for slot {slot} (found {total_obs}, need > {window_size})")
            continue

        # Chronological split index
        split_idx = int(total_obs * train_ratio)

        train_series = occupancy_values[:split_idx]
        test_series = occupancy_values[split_idx:]

        # Build train sequences
        for i in range(window_size, len(train_series)):
            X_train_list.append(train_series[i - window_size:i])
            y_train_list.append(train_series[i])

        # Build test sequences
        # Note: To predict the first test observation without looking ahead,
        # we can prime the first test window with the last (window_size) points from train_series
        combined_test_input = np.concatenate([train_series[-window_size:], test_series])
        for i in range(window_size, len(combined_test_input)):
            X_test_list.append(combined_test_input[i - window_size:i])
            y_test_list.append(combined_test_input[i])

    # Convert to NumPy arrays and reshape for LSTM: (num_samples, timesteps, features)
    X_train = np.array(X_train_list, dtype=np.float32)[:, :, np.newaxis]
    y_train = np.array(y_train_list, dtype=np.float32)

    X_test = np.array(X_test_list, dtype=np.float32)[:, :, np.newaxis]
    y_test = np.array(y_test_list, dtype=np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def preprocess_dataset(
    csv_path: str = DEFAULT_CSV_PATH,
    output_npz: str = DEFAULT_PROCESSED_NPZ,
    output_csv: str = DEFAULT_PROCESSED_CSV,
    window_size: int = 8,
    train_ratio: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Main preprocessing pipeline: loads data, creates sequences, saves npz and summary csv.
    """
    print(f"[PREPROCESS] Loading raw data from: {csv_path}")
    df = load_raw_data(csv_path)
    print(f"[PREPROCESS] Loaded {len(df)} rows across slots: {df['slot_id'].unique().tolist()}")

    # Save enriched DataFrame with time features to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[PREPROCESS] Enriched feature data saved to: {output_csv}")

    # Build sequences
    data_dict = create_sliding_window_sequences(df, window_size=window_size, train_ratio=train_ratio)

    # Save to NPZ file
    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        X_train=data_dict["X_train"],
        y_train=data_dict["y_train"],
        X_test=data_dict["X_test"],
        y_test=data_dict["y_test"],
        window_size=window_size,
        train_ratio=train_ratio
    )
    print(f"[PREPROCESS] Processed arrays saved to: {output_npz}")
    print(f"[PREPROCESS] Shapes -> X_train: {data_dict['X_train'].shape}, y_train: {data_dict['y_train'].shape}")
    print(f"[PREPROCESS] Shapes -> X_test:  {data_dict['X_test'].shape}, y_test:  {data_dict['y_test'].shape}")

    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Preprocess parking data for LSTM training")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to input raw CSV")
    parser.add_argument("--output-npz", type=str, default=DEFAULT_PROCESSED_NPZ, help="Path to output NPZ")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_PROCESSED_CSV, help="Path to output CSV")
    parser.add_argument("--window", type=int, default=8, help="Sliding window sequence size (default: 8)")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split ratio (default: 0.8)")

    args = parser.parse_args()
    preprocess_dataset(
        csv_path=args.csv,
        output_npz=args.output_npz,
        output_csv=args.output_csv,
        window_size=args.window,
        train_ratio=args.split
    )


if __name__ == "__main__":
    main()
