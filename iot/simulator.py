"""
IoT Parking Slot Simulator
--------------------------
Simulates ultrasonic sensor (HC-SR04) and occupancy data for parking slots (P1 to P4).
Outputs data to 'data/parking_data.csv' and optionally transmits readings to FastAPI backend.

In a real deployment, this module will be replaced by physical ESP32 microcontrollers
reading HC-SR04 ultrasonic distance sensors.

Sensor Distance Logic:
- Distance < 20 cm: Slot is OCCUPIED (occupied = 1) (vehicle is parked above sensor)
- Distance >= 20 cm: Slot is EMPTY (occupied = 0) (no vehicle detected)
"""

import os
import csv
import random
import argparse
from datetime import datetime, timedelta

# Configuration constants
SLOTS = ["P1", "P2", "P3", "P4"]
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data.csv")
DEFAULT_API_URL = "http://127.0.0.1:8000/api/sensor/update"

# Thresholds for realistic ultrasonic measurements (in cm)
OCCUPIED_DISTANCE_RANGE = (5, 18)     # Car present close to sensor
EMPTY_DISTANCE_RANGE = (35, 75)       # Clear space to floor/ceiling


def generate_distance(occupied: int) -> int:
    """Generate realistic ultrasonic distance in cm based on occupancy state."""
    if occupied == 1:
        return random.randint(*OCCUPIED_DISTANCE_RANGE)
    else:
        return random.randint(*EMPTY_DISTANCE_RANGE)


def get_occupancy_probability(hour: int) -> float:
    """
    Returns the probability of a slot being occupied based on the hour of the day.
    Simulates peak campus/office hours (9 AM - 6 PM) vs off-peak hours.
    """
    if 9 <= hour <= 12:
        return 0.80  # Morning peak
    elif 13 <= hour <= 17:
        return 0.70  # Afternoon busy
    elif 18 <= hour <= 21:
        return 0.40  # Evening drop-off
    elif 7 <= hour <= 8:
        return 0.35  # Early arrivals
    else:
        return 0.10  # Night / early morning low occupancy


def send_reading_to_api(reading: dict, api_url: str = DEFAULT_API_URL) -> bool:
    """
    Sends a single sensor reading to the FastAPI backend.
    Catches connection errors gracefully so simulator does not crash if server is offline.
    """
    try:
        import requests
        response = requests.post(api_url, json=reading, timeout=3.0)
        if response.status_code in (200, 201):
            return True
        else:
            print(f"[API ERROR] HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[API WARNING] Failed to connect to API ({api_url}): {e}")
        return False


def generate_batch_dataset(
    days: int = 7,
    interval_minutes: int = 15,
    output_file: str = DEFAULT_OUTPUT_PATH,
    send_api: bool = False,
    api_url: str = DEFAULT_API_URL,
):
    """
    Generates historical time-series parking sensor data over a given number of days.
    Simulates continuous parking durations for realism.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Calculate time range ending at current hour
    end_time = datetime.now().replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)
    
    records = []
    current_time = start_time
    
    # State tracking to maintain vehicle stay duration (slots don't flip randomly every step)
    slot_states = {slot: {"occupied": 0, "remaining_steps": 0} for slot in SLOTS}
    
    while current_time <= end_time:
        hour = current_time.hour
        prob = get_occupancy_probability(hour)
        
        for slot in SLOTS:
            state = slot_states[slot]
            
            # If the current stay or vacancy duration has expired, decide new state
            if state["remaining_steps"] <= 0:
                is_occupied = 1 if random.random() < prob else 0
                state["occupied"] = is_occupied
                # Stay duration between 30 mins to 3 hours (in intervals)
                state["remaining_steps"] = random.randint(2, 12)
            else:
                state["remaining_steps"] -= 1
            
            occupied = state["occupied"]
            distance_cm = generate_distance(occupied)
            
            row = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "slot_id": slot,
                "occupied": occupied,
                "distance_cm": distance_cm
            }
            records.append(row)
            
            if send_api:
                send_reading_to_api(row, api_url)
            
        current_time += timedelta(minutes=interval_minutes)
    
    # Write to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "slot_id", "occupied", "distance_cm"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"[SUCCESS] Generated {len(records)} records for slots {SLOTS} over {days} days.")
    print(f"[FILE] Saved to: {os.path.abspath(output_file)}")


def simulate_live_stream(
    output_file: str = DEFAULT_OUTPUT_PATH,
    interval_seconds: int = 3,
    send_api: bool = False,
    api_url: str = DEFAULT_API_URL,
):
    """
    Simulates live IoT sensor streaming. Appends new readings every interval_seconds.
    Optionally pushes readings to the backend API via HTTP POST.
    Press Ctrl+C to stop.
    """
    import time
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    file_exists = os.path.exists(output_file)
    
    print(f"[LIVE MODE] Streaming sensor readings every {interval_seconds}s to {output_file}...")
    if send_api:
        print(f"[HTTP POST] Live readings will be pushed to {api_url}")
    print("Press Ctrl+C to stop.\n")
    
    # Initial state
    slot_states = {slot: random.choice([0, 1]) for slot in SLOTS}
    
    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_rows = []
            
            # Occasionally toggle a slot status (15% chance per slot per tick)
            for slot in SLOTS:
                if random.random() < 0.15:
                    slot_states[slot] = 1 - slot_states[slot]
                    
                occupied = slot_states[slot]
                distance_cm = generate_distance(occupied)
                row = {
                    "timestamp": now_str,
                    "slot_id": slot,
                    "occupied": occupied,
                    "distance_cm": distance_cm
                }
                new_rows.append(row)
                status_str = "OCCUPIED" if occupied == 1 else "EMPTY"
                print(f"[{now_str}] Slot: {slot} | Status: {status_str:<8} | Distance: {distance_cm} cm")
                
                # Push to HTTP API if enabled
                if send_api:
                    sent = send_reading_to_api(row, api_url)
                    if sent:
                        print(f"  |--> [API PUSHED] Slot {slot} updated successfully")
            
            # Append rows to CSV
            with open(output_file, mode="a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["timestamp", "slot_id", "occupied", "distance_cm"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(new_rows)
                
            print("-" * 50)
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n[STOPPED] Live IoT streaming stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Smart Parking IoT Sensor Simulator")
    parser.add_argument("--days", type=int, default=7, help="Number of past days to simulate (default: 7)")
    parser.add_argument("--interval", type=int, default=15, help="Time interval between readings in minutes (default: 15)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--live", action="store_true", help="Run in continuous live streaming mode")
    parser.add_argument("--live-interval", type=int, default=3, help="Interval in seconds for live mode (default: 3)")
    parser.add_argument("--send-api", action="store_true", help="Transmit readings via HTTP POST to FastAPI backend")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Target API URL for sensor updates")
    
    args = parser.parse_args()
    
    if args.live:
        simulate_live_stream(
            output_file=args.output,
            interval_seconds=args.live_interval,
            send_api=args.send_api,
            api_url=args.api_url,
        )
    else:
        generate_batch_dataset(
            days=args.days,
            interval_minutes=args.interval,
            output_file=args.output,
            send_api=args.send_api,
            api_url=args.api_url,
        )


if __name__ == "__main__":
    main()
