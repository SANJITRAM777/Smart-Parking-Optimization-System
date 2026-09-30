"""
LSTM Model Training and Evaluation Module
----------------------------------------
Builds, trains, and evaluates a simple LSTM neural network to predict
parking slot occupancy from chronological sequence data.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Paths
DEFAULT_NPZ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.npz")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "parking_lstm.keras")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "metrics.json")


def load_data(npz_path: str = DEFAULT_NPZ_PATH):
    """Loads preprocessed training and testing arrays."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at '{npz_path}'. Run 'dnn/preprocess.py' first."
        )
    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    window_size = int(data["window_size"]) if "window_size" in data else X_train.shape[1]
    return X_train, y_train, X_test, y_test, window_size


def build_lstm_model(window_size: int = 8, num_features: int = 1):
    """
    Builds a clean, explainable LSTM architecture:
      Input(shape=(window_size, 1))
      -> LSTM(32)
      -> Dropout(0.2)
      -> Dense(16, activation='relu')
      -> Dense(1, activation='sigmoid')
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

    model = Sequential([
        Input(shape=(window_size, num_features), name="occupancy_sequence_input"),
        LSTM(32, activation="tanh", name="lstm_layer"),
        Dropout(0.2, name="dropout_regularization"),
        Dense(16, activation="relu", name="dense_feature_layer"),
        Dense(1, activation="sigmoid", name="occupancy_probability_output")
    ], name="SmartParking_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(
    npz_path: str = DEFAULT_NPZ_PATH,
    model_save_path: str = MODEL_SAVE_PATH,
    metrics_save_path: str = METRICS_SAVE_PATH,
    epochs: int = 25,
    batch_size: int = 32,
    threshold: float = 0.5
):
    """Trains the LSTM with early stopping and evaluates performance on chronological test data."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    print("[TRAIN] Loading chronological datasets...")
    X_train, y_train, X_test, y_test, window_size = load_data(npz_path)
    print(f"[TRAIN] Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples (Window size: {window_size})")

    # Build model
    model = build_lstm_model(window_size=window_size, num_features=X_train.shape[2])
    model.summary()

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\n[TRAIN] Training LSTM for up to {epochs} epochs (Batch size: {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save(model_save_path)
    print(f"\n[SAVE] Model saved successfully to: {os.path.abspath(model_save_path)}")

    # Evaluate on Chronological Test Set
    print("\n" + "=" * 55)
    print("           MODEL TEST EVALUATION RESULTS")
    print("=" * 55)

    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred_classes = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred_classes))
    prec = float(precision_score(y_test, y_pred_classes, zero_division=0))
    rec = float(recall_score(y_test, y_pred_classes, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_classes, zero_division=0))
    cm = confusion_matrix(y_test, y_pred_classes).tolist()

    print(f"Test Accuracy:     {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Test Precision:    {prec:.4f}")
    print(f"Test Recall:       {rec:.4f}")
    print(f"Test F1-Score:     {f1:.4f}")
    print(f"\nConfusion Matrix (Rows: Actual [0, 1], Cols: Predicted [0, 1]):")
    print(f"  [TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"  [FN={cm[1][0]}, TP={cm[1][1]}]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=["Available (0)", "Occupied (1)"], zero_division=0))
    print("=" * 55)

    # Save metrics JSON
    metrics = {
        "dataset_samples": int(X_train.shape[0] + X_test.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "window_size": int(window_size),
        "epochs_trained": len(history.history["loss"]),
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SAVE] Evaluation metrics saved to: {os.path.abspath(metrics_save_path)}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model for parking occupancy")
    parser.add_argument("--npz", type=str, default=DEFAULT_NPZ_PATH, help="Path to processed data NPZ")
    parser.add_argument("--model-out", type=str, default=MODEL_SAVE_PATH, help="Path to save model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")

    args = parser.parse_args()
    train_and_evaluate(
        npz_path=args.npz,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
