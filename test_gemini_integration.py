"""
Test Suite for Google Gemini Integration in GenAI Parking Assistant
-------------------------------------------------------------------
Verifies:
1. Backend works seamlessly without GEMINI_API_KEY.
2. Fallback mode is used and logged gracefully.
3. Prompt builder incorporates all required facts (slots, distances, risk, stats, recommendation).
4. System does NOT crash when API key is invalid or API call fails.
5. Optimizer remains the sole authority for slot recommendations.
6. Successful Gemini API call integration (tested via mock client).
7. API key is never exposed or returned in API responses.
"""

import os
import sys
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimization.slot_optimizer import optimize_parking_slots
from genai.parking_assistant import (
    ask_parking_assistant,
    get_api_key,
    build_gemini_prompt,
    call_gemini_api,
)


def test_no_key_fallback():
    print("[TEST 1] Testing Assistant without GEMINI_API_KEY (Fallback Mode)...")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GENAI_API_KEY": ""}, clear=False):
        res = ask_parking_assistant("Where should I park?", force_mock=True)
        assert res["mode"] == "mock_template", f"Expected mock_template mode, got {res['mode']}"
        assert res["response"], "Response should not be empty"
        assert res["recommended_slot"], "Recommended slot should be present"
        assert "GEMINI_API_KEY" not in res, "API key must not be in response dictionary"
    print(f"  --> Passed: Fallback responded in [{res['mode'].upper()}] mode: '{res['response'][:60]}...'")


def test_structured_prompt_builder():
    print("\n[TEST 2] Testing Structured Prompt Builder Content...")
    context = optimize_parking_slots()
    prompt = build_gemini_prompt("Where should I park?", context)

    # Verify all required facts are present
    assert "CRITICAL RULES" in prompt
    assert "Summary Statistics" in prompt
    assert "Slot Details" in prompt
    assert "P1" in prompt and "P2" in prompt
    assert "Authoritative Optimizer Recommended Slot" in prompt
    assert "Walking Distance to Entrance" in prompt
    assert "Predicted Future Occupancy Probability" in prompt
    assert 'USER QUESTION: "Where should I park?"' in prompt
    print("  --> Passed: Prompt contains complete telemetry, LSTM risks, and optimizer recommendation.")


def test_api_failure_graceful_fallback():
    print("\n[TEST 3] Testing Graceful Fallback on Gemini API Failure...")
    context = optimize_parking_slots()
    with patch("genai.parking_assistant.call_gemini_api", return_value=None):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "invalid_test_key"}):
            res = ask_parking_assistant("Which parking slot is closest?", optimizer_result=context)
            assert res["mode"] == "unavailable", f"Expected unavailable mode on API failure, got {res['mode']}"
            assert "temporarily unavailable" in res["response"].lower(), f"Expected service unavailable message, got: {res['response']}"
            assert res["recommended_slot"] == context["recommended_slot"]
    print("  --> Passed: API failure returned clear service unavailable error without fake responses.")


def test_gemini_successful_mock_call():
    print("\n[TEST 4] Testing Successful Gemini API Integration Flow...")
    mock_gemini_text = "I recommend parking in Slot P1. It is currently available with minimal occupancy risk."
    
    with patch("genai.parking_assistant.call_gemini_api", return_value=mock_gemini_text):
        with patch("genai.parking_assistant.get_api_key", return_value="test_valid_dummy_key"):
            res = ask_parking_assistant("Where should I park?")
            assert res["mode"] == "gemini_api", f"Expected gemini_api mode, got {res['mode']}"
            assert res["response"] == mock_gemini_text
            assert res["recommended_slot"] is not None
            assert "test_valid_dummy_key" not in str(res), "API key must NEVER be leaked in results"
    print(f"  --> Passed: Gemini API mode activated [{res['mode'].upper()}]: '{res['response']}'")


def test_optimizer_sole_authority():
    print("\n[TEST 5] Testing that Optimizer remains the Single Decision-Maker...")
    custom_status = {"P1": "occupied", "P2": "occupied", "P3": "available", "P4": "occupied"}
    opt_res = optimize_parking_slots(slot_statuses=custom_status)
    assert opt_res["recommended_slot"] == "P3", "Optimizer must select P3"

    prompt = build_gemini_prompt("Where should I park?", opt_res)
    assert "Authoritative Optimizer Recommended Slot: P3" in prompt
    print("  --> Passed: Optimizer decision strictly governs the prompt context.")


def test_security_no_key_exposure():
    print("\n[TEST 6] Testing Security and Key Confidentiality...")
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
    assert os.path.exists(gitignore_path)
    with open(gitignore_path, "r", encoding="utf-8") as f:
        git_content = f.read()
    assert ".env" in git_content, ".env must be in .gitignore"
    print("  --> Passed: .env is strictly ignored in .gitignore.")


def run_all():
    print("==================================================")
    print("    RUNNING GEMINI API INTEGRATION TEST SUITE")
    print("==================================================")
    test_no_key_fallback()
    test_structured_prompt_builder()
    test_api_failure_graceful_fallback()
    test_gemini_successful_mock_call()
    test_optimizer_sole_authority()
    test_security_no_key_exposure()
    print("\n" + "=" * 50)
    print("   ALL GEMINI INTEGRATION TESTS PASSED (6/6)!")
    print("==================================================\n")


if __name__ == "__main__":
    run_all()
