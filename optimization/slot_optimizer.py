"""
Parking Slot Optimization Engine
--------------------------------
Calculates utility scores and recommends the optimal parking bay based on:
1. Real-time physical availability (from SQLite)
2. DNN-predicted future occupancy risk (from LSTM model)
3. Walking distance / proximity to the entrance

Design Rule: Only currently AVAILABLE slots can be recommended.
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fixed walking distances in meters from parking slot to building entrance
# These are baseline demonstration values that can be updated for different facility layouts.
SLOT_DISTANCES_METERS = {
    "P1": 10.0,
    "P2": 25.0,
    "P3": 40.0,
    "P4": 15.0,
}

# Reference maximum distance for score normalization
MAX_REFERENCE_DISTANCE = 50.0


def compute_proximity_score(distance_m: float, max_dist: float = MAX_REFERENCE_DISTANCE) -> float:
    """
    Normalizes walking distance into a proximity score in range [0.0, 1.0].
    Shorter distance yields a higher proximity score.
    """
    clipped_dist = min(max(distance_m, 0.0), max_dist)
    return 1.0 - (clipped_dist / max_dist)


def compute_slot_utility(
    distance_m: float,
    predicted_occupancy_prob: float,
    weight_distance: float = 0.5,
    weight_prediction: float = 0.5,
) -> float:
    """
    Computes a combined utility score between 0.0 and 1.0.
    
    Formula:
      Score = w_dist * ProximityScore + w_pred * (1.0 - PredictedOccupancyRisk)
      
    Both sub-scores range from 0 (worst) to 1 (best).
    """
    proximity_score = compute_proximity_score(distance_m)
    availability_score = 1.0 - predicted_occupancy_prob
    
    utility = (weight_distance * proximity_score) + (weight_prediction * availability_score)
    return round(utility, 4)


def get_current_system_state() -> Dict[str, Any]:
    """
    Fetches live slot states from SQLite database and LSTM occupancy predictions.
    Falls back gracefully if database or DNN model is not yet loaded.
    """
    slot_statuses = {}
    predictions = {}

    # 1. Fetch current slot statuses from SQLite
    try:
        from backend.database import get_all_slots, init_db
        init_db()
        db_slots = get_all_slots()
        for s in db_slots:
            slot_statuses[s["slot_id"]] = s["status"].lower()
    except Exception as e:
        # Default fallback if DB is unreachable
        print(f"[OPTIMIZER WARNING] Could not read SQLite slots ({e}). Using default 'available'.")
        slot_statuses = {k: "available" for k in SLOT_DISTANCES_METERS.keys()}

    # 2. Fetch LSTM predictions
    try:
        from dnn.predict import predict_all_slots
        pred_dict = predict_all_slots()
        for slot_id, p_info in pred_dict.items():
            predictions[slot_id] = p_info["probability"]
    except Exception as e:
        # Default fallback if DNN model is not found
        print(f"[OPTIMIZER WARNING] Could not run LSTM predict ({e}). Using default probability 0.20.")
        predictions = {k: 0.20 for k in SLOT_DISTANCES_METERS.keys()}

    return {
        "slot_statuses": slot_statuses,
        "predictions": predictions,
    }


def optimize_parking_slots(
    slot_statuses: Optional[Dict[str, str]] = None,
    predictions: Optional[Dict[str, float]] = None,
    distances: Optional[Dict[str, float]] = None,
    weight_distance: float = 0.5,
    weight_prediction: float = 0.5,
) -> Dict[str, Any]:
    """
    Core Optimization Algorithm:
    - Filters for slots currently 'available'.
    - Ranks available slots using proximity and predicted availability.
    - Selects the highest scoring slot as 'recommended_slot'.
    """
    if distances is None:
        distances = SLOT_DISTANCES_METERS

    # If status or predictions were not passed explicitly, retrieve live system state
    if slot_statuses is None or predictions is None:
        state = get_current_system_state()
        if slot_statuses is None:
            slot_statuses = state["slot_statuses"]
        if predictions is None:
            predictions = state["predictions"]

    available_candidates = []
    all_slots_summary = []

    for slot_id, dist_m in distances.items():
        status = slot_statuses.get(slot_id, "available").lower()
        pred_occ = predictions.get(slot_id, 0.20)
        
        is_available = (status == "available")
        score = compute_slot_utility(
            distance_m=dist_m,
            predicted_occupancy_prob=pred_occ,
            weight_distance=weight_distance,
            weight_prediction=weight_prediction,
        ) if is_available else 0.0

        slot_info = {
            "slot_id": slot_id,
            "status": status,
            "distance_m": dist_m,
            "predicted_occupancy": round(pred_occ, 4),
            "score": score,
        }
        all_slots_summary.append(slot_info)

        if is_available:
            available_candidates.append(slot_info)

    # Sort available slots by utility score descending
    available_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Handle scenario: All slots occupied
    if not available_candidates:
        return {
            "recommended_slot": None,
            "score": 0.0,
            "reason": "All parking slots are currently occupied. Please wait for a vehicle to depart.",
            "distance_m": None,
            "predicted_occupancy": None,
            "ranked_slots": [],
            "all_slots": all_slots_summary,
        }

    # Best recommended slot
    best = available_candidates[0]
    reason = (
        f"Slot {best['slot_id']} is currently available, has low predicted occupancy "
        f"risk ({best['predicted_occupancy'] * 100:.1f}%), and is only "
        f"{int(best['distance_m'])} meters from the entrance."
    )

    return {
        "recommended_slot": best["slot_id"],
        "score": best["score"],
        "reason": reason,
        "distance_m": best["distance_m"],
        "predicted_occupancy": best["predicted_occupancy"],
        "ranked_slots": available_candidates,
        "all_slots": all_slots_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Parking Slot Optimizer")
    parser.add_argument("--w-dist", type=float, default=0.5, help="Weight for walking distance (default: 0.5)")
    parser.add_argument("--w-pred", type=float, default=0.5, help="Weight for predicted availability (default: 0.5)")

    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("         PARKING SLOT OPTIMIZATION RESULTS")
    print("=" * 55)

    result = optimize_parking_slots(weight_distance=args.w_dist, weight_prediction=args.w_pred)

    print(f"Recommended Slot:    {result['recommended_slot']}")
    print(f"Optimization Score:  {result['score']}")
    print(f"Reason:              {result['reason']}\n")

    print("Ranked Available Slots:")
    for rank, slot in enumerate(result["ranked_slots"], start=1):
        print(f"  #{rank} Slot {slot['slot_id']} | Score: {slot['score']:.4f} | Distance: {slot['distance_m']}m | Pred Occupancy: {slot['predicted_occupancy'] * 100:.1f}%")

    print("\nAll Slot States:")
    for s in result["all_slots"]:
        print(f"  - Slot {s['slot_id']}: {s['status'].upper():<9} (Distance: {s['distance_m']}m, Pred: {s['predicted_occupancy'] * 100:.1f}%)")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
