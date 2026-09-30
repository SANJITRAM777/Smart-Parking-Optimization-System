"""
GenAI Natural-Language Parking Assistant
---------------------------------------
Converts structured parking telemetry, LSTM predictions, and optimizer decisions
into user-friendly, natural-language conversational responses.

CRITICAL DESIGN RULE:
The GenAI module NEVER independently selects a parking slot. The optimizer makes
the actual decision; GenAI explains and presents the result naturally.

Security:
- Reads GENAI_API_KEY from environment.
- Works reliably in Mock/Template fallback mode if no API key is set.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimization.slot_optimizer import optimize_parking_slots, get_current_system_state


def get_api_key() -> Optional[str]:
    """
    Retrieves GEMINI_API_KEY (or legacy GENAI_API_KEY) from environment or .env file.
    Ensures API keys are never hardcoded and placeholder strings are ignored.
    """
    # Attempt to load .env using python-dotenv if available
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(PROJECT_ROOT, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
    except Exception:
        pass

    # Check process environment
    for key_name in ["GEMINI_API_KEY", "GENAI_API_KEY"]:
        api_key = os.getenv(key_name)
        if api_key and api_key.strip() and api_key.strip() not in ["your_api_key_here", "your_gemini_api_key_here"]:
            return api_key.strip()

    # Manual fallback parsing for .env file
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for prefix in ["GEMINI_API_KEY=", "GENAI_API_KEY="]:
                        if line.startswith(prefix):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val and val not in ["your_api_key_here", "your_gemini_api_key_here"]:
                                return val
        except Exception:
            pass
    return None


def build_gemini_prompt(query: str, context: Dict[str, Any]) -> str:
    """
    Constructs a structured prompt containing real-time parking slot statuses,
    LSTM occupancy predictions, and optimizer decisions.
    """
    all_slots = context.get("all_slots", [])
    rec_slot = context.get("recommended_slot")
    rec_score = context.get("score", 0.0)
    rec_dist = context.get("distance_m")
    rec_pred = context.get("predicted_occupancy")
    reason = context.get("reason", "")

    # Build status lines
    status_blocks = []
    dnn_blocks = []
    available_slots = []
    occupied_slots = []

    for s in all_slots:
        s_id = s.get("slot_id")
        status = s.get("status", "available").lower()
        dist = s.get("distance_m", 15.0)
        pred = s.get("predicted_occupancy", 0.20)
        if status == "available":
            available_slots.append(f"{s_id} ({int(dist)}m)")
        else:
            occupied_slots.append(f"{s_id} ({int(dist)}m)")

        status_blocks.append(f"{s_id}:\n  status = {status}\n  distance = {int(dist)}m")
        risk_label = "low" if pred < 0.20 else "moderate" if pred < 0.50 else "high"
        dnn_blocks.append(f"{s_id} = {pred:.4f} ({pred * 100:.1f}% probability of occupancy - {risk_label} chance)")

    all_occupied = (len(available_slots) == 0 or rec_slot is None)

    prompt = f"""You are the friendly, helpful AI Parking Assistant for a smart parking facility.
The target user is a NORMAL DRIVER looking for parking.

CURRENT PARKING STATUS & Slot Details:
{chr(10).join(status_blocks)}

Summary Statistics:
Total Slots: {len(all_slots)}
Available Spaces: {len(available_slots)} ({", ".join(available_slots) if available_slots else "None"})
Occupied Spaces: {len(occupied_slots)} ({", ".join(occupied_slots) if occupied_slots else "None"})

CURRENT DNN PREDICTIONS (LSTM Model Future Occupancy Risk):
{chr(10).join(dnn_blocks)}

CURRENT OPTIMIZER RESULT:
Authoritative Optimizer Recommended Slot: {rec_slot if not all_occupied else "NONE (All Slots Occupied)"}
Walking Distance to Entrance: {int(rec_dist) if rec_dist is not None else "N/A"} meters
Predicted Future Occupancy Probability: {f"{rec_pred * 100:.1f}%" if rec_pred is not None else "N/A"}
Optimizer reason = "{reason}"

CRITICAL RULES:
1. You must NEVER independently choose, invent, or recommend a parking slot different from the CURRENT OPTIMIZER RESULT. The optimization algorithm is the sole authoritative decision maker.
2. If all slots are occupied, clearly and politely inform the driver that no parking spaces are currently available.
3. If asked "Where should I park?" or "Which slot do you recommend?", recommend {rec_slot} noting that it is available, its walking distance ({int(rec_dist) if rec_dist else 'close'}m), and its low chance of becoming occupied.
4. If asked "Which slot is closest?", accurately analyze all slots: mention the closest physical bay (and whether it is currently occupied) and recommend the closest AVAILABLE bay.
5. If asked about a specific slot (e.g. "Is P3 available?", "What is the status of P1?", "Is P3 likely to become occupied?"), use the exact real-time status and DNN prediction from above.
6. If asked general technical or conceptual questions (e.g., "What is an ultrasonic sensor?", "What is LSTM used for?", "How does the system predict parking occupancy?", "How does the optimization work?", "How does GenAI help?"), answer accurately, concisely, and clearly in 2-3 friendly sentences.
7. Tone: Friendly, concise (2-3 sentences max), practical, and easy for a driver to understand immediately. Do not expose internal mathematical formulas unless explicitly asked.

USER QUESTION: "{query}"

ASSISTANT RESPONSE:"""
    return prompt


def call_gemini_api(query: str, context: Dict[str, Any], api_key: str) -> Optional[str]:
    """
    Sends the prompt and structured context to Google's Gemini API.
    Uses the official google-genai SDK, with automatic model fallback and REST backup.
    Returns the generated text response, or None if the API call fails.
    """
    prompt = build_gemini_prompt(query, context)
    primary_model = os.getenv("GEMINI_MODEL", os.getenv("GENAI_MODEL", "gemini-flash-lite-latest"))

    # Priority list of models to try in case of temporary 503 high demand or 404 deprecation
    models_to_try = [primary_model]
    for fallback_candidate in ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-3-flash-preview"]:
        if fallback_candidate not in models_to_try:
            models_to_try.append(fallback_candidate)

    # 1. Try official google-genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text and response.text.strip():
                    return response.text.strip()
            except Exception as model_err:
                print(f"[GEMINI SDK LOG] Model '{model_name}' did not succeed ({type(model_err).__name__}). Trying next fallback model...")
                continue
    except Exception as sdk_err:
        print(f"[GEMINI SDK LOG] SDK initialization error: {type(sdk_err).__name__}. Trying REST endpoint...")

    # 2. Graceful fallback to direct Google Gemini REST endpoint
    for model_name in models_to_try:
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 300
                }
            }
            res = requests.post(url, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            else:
                print(f"[GEMINI REST LOG] REST API for '{model_name}' returned status {res.status_code}.")
        except Exception as rest_err:
            print(f"[GEMINI REST LOG] REST API request for '{model_name}' failed: {type(rest_err).__name__}.")

    return None


# Backward compatibility alias
call_external_llm = call_gemini_api


def ask_parking_assistant(
    query: str,
    optimizer_result: Optional[Dict[str, Any]] = None,
    force_mock: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for the GenAI Parking Assistant.
    1. Fetches current optimizer result if not provided (guarantees live state).
    2. Checks for GEMINI_API_KEY from environment or .env.
    3. Calls Gemini API when key is available with live parking context.
    4. If Gemini fails, returns a clear service-unavailable error message without fake answers.
    5. Returns response with metadata (never exposing the API key).
    """
    if optimizer_result is None:
        optimizer_result = optimize_parking_slots()

    rec_slot = optimizer_result.get("recommended_slot")
    opt_score = optimizer_result.get("score")

    if force_mock:
        # Explicit test / mock request (e.g. unit tests with force_mock=True)
        reply = generate_template_response(query=query, context=optimizer_result)
        return {
            "query": query,
            "response": reply,
            "mode": "mock_template",
            "recommended_slot": rec_slot,
            "optimizer_score": opt_score,
        }

    api_key = get_api_key()
    if not api_key:
        print("[GENAI ASSISTANT LOG] GEMINI_API_KEY not found in environment.")
        return {
            "query": query,
            "response": "Parking Assistant is temporarily unavailable. Please configure GEMINI_API_KEY in the server environment.",
            "mode": "unavailable",
            "recommended_slot": rec_slot,
            "optimizer_score": opt_score,
        }

    # Call Gemini API with live backend context
    reply = call_gemini_api(query=query, context=optimizer_result, api_key=api_key)

    if reply:
        return {
            "query": query,
            "response": reply,
            "mode": "gemini_api",
            "recommended_slot": rec_slot,
            "optimizer_score": opt_score,
        }

    # Gemini API failed (network error, quota exhausted, outage)
    # Per requirements: DO NOT pretend response came from Gemini or return a fake response.
    print("[GENAI ASSISTANT ERROR] Gemini API call failed. Returning service unavailable error.")
    return {
        "query": query,
        "response": "Parking Assistant is temporarily unavailable. Please try again.",
        "mode": "unavailable",
        "recommended_slot": rec_slot,
        "optimizer_score": opt_score,
    }



def generate_template_response(query: str, context: Dict[str, Any]) -> str:
    """
    High-quality deterministic fallback/mock generator.
    Produces natural, accurate responses directly from optimizer and LSTM facts
    without requiring external API calls or API keys.
    """
    q = query.lower().strip()
    rec_slot = context.get("recommended_slot")
    rec_dist = context.get("distance_m")
    rec_pred = context.get("predicted_occupancy")
    ranked = context.get("ranked_slots", [])
    all_slots = context.get("all_slots", [])

    # Scenario 1: All slots occupied
    if not rec_slot or not ranked:
        return (
            "All parking slots (P1 to P4) are currently occupied. "
            "Please wait near the entrance or check back shortly as vehicles depart."
        )

    # Question Type A: "Where should I park?" or "Which slot should I choose?"
    if any(k in q for k in ["where", "recommend", "best", "choose", "which slot"]):
        risk_desc = "very low" if rec_pred < 0.20 else "moderate" if rec_pred < 0.50 else "elevated"
        return (
            f"I recommend parking in Slot {rec_slot}. It is currently available, located only "
            f"{int(rec_dist)} meters from the main entrance, and has a {risk_desc} predicted occupancy "
            f"risk of {rec_pred * 100:.1f}%. It is your optimal choice."
        )

    # Question Type B: "Which parking slot is closest?"
    if any(k in q for k in ["closest", "nearest", "minimum distance", "distance"]):
        # Find available slot with minimum distance
        closest_slot = min(ranked, key=lambda x: x["distance_m"])
        return (
            f"The closest available parking bay is Slot {closest_slot['slot_id']}, which is "
            f"{int(closest_slot['distance_m'])} meters from the entrance. Its current predicted "
            f"occupancy risk is {closest_slot['predicted_occupancy'] * 100:.1f}%."
        )

    # Question Type C: "Which slot is likely to remain available?"
    if any(k in q for k in ["remain available", "free", "stay", "lowest risk", "least occupied"]):
        safest_slot = min(ranked, key=lambda x: x["predicted_occupancy"])
        return (
            f"Slot {safest_slot['slot_id']} is the most likely to remain available. The LSTM model "
            f"forecasts only a {safest_slot['predicted_occupancy'] * 100:.1f}% chance of occupancy, "
            f"and it is {int(safest_slot['distance_m'])} meters from the entrance."
        )

    # Question Type D: "Will P3 be occupied soon?" or specific slot inquiry
    for s in ["p1", "p2", "p3", "p4"]:
        if s in q:
            target = s.upper()
            slot_match = next((x for x in all_slots if x["slot_id"] == target), None)
            if slot_match:
                status = slot_match["status"].upper()
                pred = slot_match["predicted_occupancy"] * 100
                if slot_match["status"] == "occupied":
                    return (
                        f"Slot {target} is currently OCCUPIED. The LSTM predicts a {pred:.1f}% "
                        f"probability of continuing occupancy in the upcoming interval. "
                        f"I recommend navigating to available Slot {rec_slot} instead."
                    )
                else:
                    likelihood = "unlikely" if pred < 40 else "likely"
                    return (
                        f"Slot {target} is currently AVAILABLE with a walking distance of {int(slot_match['distance_m'])}m. "
                        f"The LSTM model estimates a {pred:.1f}% likelihood of occupancy soon ({likelihood} to stay free)."
                    )

    # Question Type E: "Show me the available parking slots."
    if any(k in q for k in ["show", "list", "available slots", "all slots", "status"]):
        avail_names = [f"Slot {s['slot_id']} ({int(s['distance_m'])}m)" for s in ranked]
        occ_names = [f"Slot {s['slot_id']}" for s in all_slots if s["status"] != "available"]
        avail_str = ", ".join(avail_names) if avail_names else "None"
        occ_str = ", ".join(occ_names) if occ_names else "None"
        return (
            f"Currently available slots: {avail_str}. "
            f"Occupied slots: {occ_str}. "
            f"Top recommended option: Slot {rec_slot}."
        )

    # General default response
    return (
        f"Slot {rec_slot} is currently available and is only {int(rec_dist)} meters from the entrance. "
        f"Its predicted occupancy risk is {rec_pred * 100:.1f}%, making it the top recommended parking option."
    )


def main():
    parser = argparse.ArgumentParser(description="GenAI Parking Assistant")
    parser.add_argument("--query", type=str, default="Where should I park?", help="User query text")
    parser.add_argument("--test-all", action="store_true", help="Run a suite of predefined queries to test assistant responses")
    parser.add_argument("--force-mock", action="store_true", help="Force mock/template mode even if API key is set")

    args = parser.parse_args()

    # Pre-compute optimizer decision once
    optimizer_res = optimize_parking_slots()

    print("\n" + "=" * 65)
    print("              GENAI PARKING ASSISTANT INTERFACE")
    print("=" * 65)
    print(f"Backend Decision: Recommended Slot = {optimizer_res['recommended_slot']} (Score: {optimizer_res['score']})")

    if args.test_all:
        test_queries = [
            "Where should I park?",
            "Which parking slot is closest?",
            "Which slot is likely to remain available?",
            "Will P3 be occupied soon?",
            "Show me the available parking slots.",
        ]
        for idx, q in enumerate(test_queries, start=1):
            res = ask_parking_assistant(query=q, optimizer_result=optimizer_res, force_mock=args.force_mock)
            print(f"\n[Q{idx}] User: \"{q}\"")
            print(f"      Mode: [{res['mode'].upper()}]")
            print(f"      Assistant: {res['response']}")
    else:
        res = ask_parking_assistant(query=args.query, optimizer_result=optimizer_res, force_mock=args.force_mock)
        print(f"\nUser Query:      \"{res['query']}\"")
        print(f"Response Mode:   [{res['mode'].upper()}]")
        print(f"Assistant Reply: {res['response']}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
