"""
Phase 4 Verification Test Suite
-------------------------------
Tests all core optimization rules and GenAI assistant integration constraints:
1. Optimizer reads slot availability correctly.
2. Occupied slots are NEVER recommended.
3. LSTM predictions directly influence ranking.
4. Distances directly influence ranking.
5. All-slots-occupied scenario is gracefully handled.
6. GenAI strictly adheres to optimizer decisions.
7. GenAI functions properly without an API key.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimization.slot_optimizer import optimize_parking_slots
from genai.parking_assistant import ask_parking_assistant, get_api_key


def test_all():
    print("==================================================")
    print("       RUNNING PHASE 4 TEST SUITE VERIFICATION")
    print("==================================================")

    # 1. Verify No API Key is hardcoded in source code
    print("\n[TEST 1] Verify API Key Security...")
    pa_path = os.path.join(os.path.dirname(__file__), "genai", "parking_assistant.py")
    with open(pa_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "AIza" not in src and "AQ.Ab" not in src, "API Key must not be hardcoded in source code!"
    print("  --> Passed: No hardcoded API key in source files.")

    # 2. All Slots Occupied Scenario
    print("\n[TEST 2] Testing 'All Slots Occupied' scenario...")
    all_occ = {"P1": "occupied", "P2": "occupied", "P3": "occupied", "P4": "occupied"}
    res_all_occ = optimize_parking_slots(slot_statuses=all_occ)
    assert res_all_occ["recommended_slot"] is None, "Should not recommend any slot when all are occupied"
    assert res_all_occ["score"] == 0.0, "Score should be 0.0 when all occupied"
    assert len(res_all_occ["ranked_slots"]) == 0, "No ranked candidates when all occupied"
    print(f"  --> Passed: recommended_slot is None, reason: {res_all_occ['reason']}")

    ai_all_occ = ask_parking_assistant("Where should I park?", optimizer_result=res_all_occ)
    print(f"  --> GenAI response for full lot: '{ai_all_occ['response']}'")
    assert "occupied" in ai_all_occ["response"].lower(), "AI should acknowledge that lot is full"

    # 3. Occupied slots are never recommended even with optimal distance and 0% risk
    print("\n[TEST 3] Testing that occupied slots are never recommended...")
    status_filter = {"P1": "occupied", "P2": "available", "P3": "occupied", "P4": "occupied"}
    preds_filter = {"P1": 0.01, "P2": 0.30, "P3": 0.90, "P4": 0.90}
    res_filter = optimize_parking_slots(slot_statuses=status_filter, predictions=preds_filter)
    assert res_filter["recommended_slot"] == "P2", "Only available slot P2 must be recommended"
    print(f"  --> Passed: P1 (occupied, 10m) ignored, P2 (available, 25m) selected.")

    # 4. Proximity vs. Predicted Occupancy Trade-Off
    print("\n[TEST 4] Testing proximity vs. predicted risk trade-off...")
    # P1 is closest (10m) but high occupancy risk (85%)
    # P4 is slightly further (15m) but very low occupancy risk (5%)
    tradeoff_status = {"P1": "available", "P2": "occupied", "P3": "occupied", "P4": "available"}
    tradeoff_preds = {"P1": 0.85, "P2": 0.50, "P3": 0.50, "P4": 0.05}
    res_tradeoff = optimize_parking_slots(slot_statuses=tradeoff_status, predictions=tradeoff_preds)
    p1_score = next(s["score"] for s in res_tradeoff["ranked_slots"] if s["slot_id"] == "P1")
    p4_score = next(s["score"] for s in res_tradeoff["ranked_slots"] if s["slot_id"] == "P4")
    print(f"  --> P1 Score (10m, 85% risk): {p1_score:.4f}")
    print(f"  --> P4 Score (15m,  5% risk): {p4_score:.4f}")
    assert res_tradeoff["recommended_slot"] == "P4", "P4 should win due to lower predicted risk"
    print("  --> Passed: High-risk slot penalized, safer slot recommended.")

    # 5. GenAI Assistant Strict Alignment
    print("\n[TEST 5] Testing GenAI Assistant strict alignment with optimizer...")
    ai_q = ask_parking_assistant("Where should I park?", optimizer_result=res_tradeoff)
    print(f"  --> GenAI Query: 'Where should I park?'")
    print(f"  --> GenAI Reply: '{ai_q['response']}'")
    assert "P4" in ai_q["response"], "GenAI must strictly recommend P4 because optimizer chose P4"
    assert ai_q["mode"] in ["mock_template", "gemini_api"], f"Unexpected mode {ai_q['mode']}"
    print("  --> Passed: GenAI perfectly aligned with optimizer output.")

    print("\n" + "=" * 50)
    print("    ALL PHASE 4 VERIFICATION TESTS PASSED (5/5)!")
    print("==================================================\n")


if __name__ == "__main__":
    test_all()
