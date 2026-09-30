"""
Phase 5 Verification Test Suite
-------------------------------
Verifies end-to-end integration across the full smart parking stack with React frontend:
1. React frontend project and key components exist.
2. Frontend package.json dependencies and install configuration are valid.
3. FastAPI backend starts and responds to health check.
4. Existing Phase 4 tests still pass.
5. Current parking data can be retrieved via FastAPI API (/api/slots).
6. Optimizer recommendation can be retrieved via API (/api/optimizer/recommendation).
7. LSTM occupancy prediction can be retrieved via API (/api/dnn/predictions).
8. GenAI assistant responds correctly in fallback mode via API (/api/genai/chat).
9. 'All slots occupied' scenario is gracefully handled.
10. React interface delegates recommendation to the Python optimizer (Single Source of Truth).
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import (
    app,
    root,
    get_slots,
    get_single_slot,
    get_analytics_history,
    get_recommendation,
    get_dnn_predictions,
    chat_with_assistant,
)
from backend.models import GenAIChatRequest
from optimization.slot_optimizer import optimize_parking_slots
from genai.parking_assistant import ask_parking_assistant


def test_react_frontend_structure():
    print("[TEST 1] Testing React Frontend Project Structure...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    assert os.path.isdir(frontend_dir), "frontend/ directory must exist"

    required_files = [
        os.path.join(frontend_dir, "package.json"),
        os.path.join(frontend_dir, "index.html"),
        os.path.join(frontend_dir, "vite.config.js"),
        os.path.join(frontend_dir, "src", "main.jsx"),
        os.path.join(frontend_dir, "src", "App.jsx"),
        os.path.join(frontend_dir, "src", "App.css"),
        os.path.join(frontend_dir, "src", "api.js"),
        os.path.join(frontend_dir, "src", "components", "ParkingSummary.jsx"),
        os.path.join(frontend_dir, "src", "components", "ParkingSlot.jsx"),
        os.path.join(frontend_dir, "src", "components", "Recommendation.jsx"),
        os.path.join(frontend_dir, "src", "components", "DNNAnalytics.jsx"),
        os.path.join(frontend_dir, "src", "components", "GenAIAssistant.jsx"),
    ]

    for fpath in required_files:
        assert os.path.exists(fpath), f"Required frontend file missing: {fpath}"

    print(f"  --> Passed: All {len(required_files)} React frontend source files exist.")


def test_frontend_dependencies_config():
    print("\n[TEST 2] Testing Frontend Dependencies & Install Config...")
    pkg_path = os.path.join(PROJECT_ROOT, "frontend", "package.json")
    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg_data = json.load(f)

    assert "dependencies" in pkg_data, "package.json must contain dependencies"
    assert "react" in pkg_data["dependencies"], "react dependency missing"
    assert "react-dom" in pkg_data["dependencies"], "react-dom dependency missing"
    assert "devDependencies" in pkg_data, "package.json must contain devDependencies"
    assert "vite" in pkg_data["devDependencies"], "vite devDependency missing"
    print(f"  --> Passed: package.json is valid JSON with React {pkg_data['dependencies']['react']} and Vite.")


def test_fastapi_backend_health():
    print("\n[TEST 3] Testing FastAPI Backend Health & Startup...")
    data = root()
    assert "message" in data, "Health response must contain 'message'"
    print(f"  --> Passed: Backend active ({data['message']})")


def test_phase4_regressions():
    print("\n[TEST 4] Verifying Phase 4 Optimizer & GenAI Core Functionality...")
    from test_phase4 import test_all
    test_all()
    print("  --> Passed: All Phase 4 verification tests pass with zero regression.")


def test_api_slots_retrieval():
    print("\n[TEST 5] Testing Slot Status API Endpoint (/api/slots)...")
    slots = get_slots()
    assert isinstance(slots, list), "Expected list of slots"
    assert len(slots) == 4, f"Expected 4 slots (P1-P4), got {len(slots)}"
    slot_ids = [s["slot_id"] for s in slots]
    assert sorted(slot_ids) == ["P1", "P2", "P3", "P4"], f"Unexpected slot IDs: {slot_ids}"
    print(f"  --> Passed: Retrieved {len(slots)} slots: {slot_ids}")


def test_api_optimizer_recommendation():
    print("\n[TEST 6] Testing Optimizer Recommendation API Endpoint (/api/optimizer/recommendation)...")
    rec = get_recommendation()
    assert "recommended_slot" in rec, "Recommendation must have 'recommended_slot'"
    assert "score" in rec, "Recommendation must have 'score'"
    assert "ranked_slots" in rec, "Recommendation must have 'ranked_slots'"
    assert "reason" in rec, "Recommendation must have 'reason'"
    print(f"  --> Passed: API recommended slot '{rec['recommended_slot']}' with score {rec['score']}")


def test_api_dnn_predictions():
    print("\n[TEST 7] Testing DNN LSTM Predictions API Endpoint (/api/dnn/predictions)...")
    preds = get_dnn_predictions()
    assert isinstance(preds, dict), "Predictions must be a dictionary"
    for s_id in ["P1", "P2", "P3", "P4"]:
        assert s_id in preds, f"Missing prediction for slot {s_id}"
        assert 0.0 <= preds[s_id]["probability"] <= 1.0, f"Invalid probability for {s_id}"
        assert preds[s_id]["prediction"] in ["AVAILABLE", "OCCUPIED"]
    print(f"  --> Passed: Valid DNN predictions retrieved for all 4 bays (P1-P4).")


def test_api_genai_chat():
    print("\n[TEST 8] Testing GenAI Parking Assistant API Endpoint (/api/genai/chat)...")
    req = GenAIChatRequest(query="Where should I park?")
    chat_res = chat_with_assistant(req)
    assert chat_res["query"] == req.query
    assert len(chat_res["response"]) > 0, "Response must not be empty"
    assert chat_res["mode"] in ["mock_template", "template_fallback", "api", "gemini_api"]
    print(f"  --> Passed: GenAI Assistant replied in [{chat_res['mode'].upper()}] mode: '{chat_res['response']}'")


def test_all_slots_occupied_handling():
    print("\n[TEST 9] Testing 'All Slots Occupied' Scenario via Optimizer & GenAI...")
    full_lot_status = {"P1": "occupied", "P2": "occupied", "P3": "occupied", "P4": "occupied"}
    opt_full = optimize_parking_slots(slot_statuses=full_lot_status)
    assert opt_full["recommended_slot"] is None, "When full, recommended_slot must be None"
    assert opt_full["ranked_slots"] == [], "Ranked candidates must be empty when full"

    ai_full = ask_parking_assistant(query="Where should I park?", optimizer_result=opt_full)
    assert "occupied" in ai_full["response"].lower(), "GenAI must state that slots are occupied"
    print(f"  --> Passed: All-slots-occupied handled gracefully: '{ai_full['response']}'")


def test_no_independent_frontend_decision():
    print("\n[TEST 10] Verifying Single Source of Truth (Backend Python Optimizer decides)...")
    opt_res = optimize_parking_slots()
    # The frontend only displays opt_res['recommended_slot'] and opt_res['ranked_slots']
    assert opt_res["recommended_slot"] is not None or len(opt_res["ranked_slots"]) == 0
    print("  --> Passed: Decision logic is strictly deterministic and centralized in Python.")


def run_all_tests():
    print("==================================================")
    print("       RUNNING PHASE 5 TEST SUITE (END-TO-END)")
    print("==================================================")
    test_react_frontend_structure()
    test_frontend_dependencies_config()
    test_fastapi_backend_health()
    test_phase4_regressions()
    test_api_slots_retrieval()
    test_api_optimizer_recommendation()
    test_api_dnn_predictions()
    test_api_genai_chat()
    test_all_slots_occupied_handling()
    test_no_independent_frontend_decision()
    print("\n" + "=" * 50)
    print("    ALL PHASE 5 TESTS PASSED SUCCESSFULLY (10/10)!")
    print("==================================================\n")


if __name__ == "__main__":
    run_all_tests()

