"""
Database Module for Smart Parking System
----------------------------------------
Manages SQLite database connections, table creation, and queries
for parking slots and sensor telemetry logs.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

# Database file location: stored under data/parking.db
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "parking.db")

# Default parking slots initialized in the system
INITIAL_SLOTS = [
    {"slot_id": "P1", "location_label": "Slot P1 - Near Entrance"},
    {"slot_id": "P2", "location_label": "Slot P2 - Ground Floor East"},
    {"slot_id": "P3", "location_label": "Slot P3 - Ground Floor West"},
    {"slot_id": "P4", "location_label": "Slot P4 - Near Exit"},
]


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with Row mapping enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn


def init_db():
    """
    Initializes the database tables:
    1. parking_slots: Stores current state of each slot (P1 - P4)
    2. sensor_logs: Stores historical sensor readings from IoT / simulator
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. parking_slots table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_slots (
            slot_id TEXT PRIMARY KEY,
            location_label TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'available',
            last_updated TEXT
        )
    """)

    # 2. sensor_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            slot_id TEXT NOT NULL,
            occupied INTEGER NOT NULL,
            distance_cm REAL NOT NULL,
            FOREIGN KEY (slot_id) REFERENCES parking_slots(slot_id)
        )
    """)

    # Populate initial parking slots if not present
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for slot in INITIAL_SLOTS:
        cursor.execute("""
            INSERT OR IGNORE INTO parking_slots (slot_id, location_label, status, last_updated)
            VALUES (?, ?, 'available', ?)
        """, (slot["slot_id"], slot["location_label"], now_str))

    conn.commit()
    conn.close()


def insert_sensor_reading(timestamp: str, slot_id: str, occupied: int, distance_cm: float) -> int:
    """
    Inserts a new sensor reading into sensor_logs and updates the slot's live status.
    Returns the inserted log record ID.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Insert into sensor_logs
    cursor.execute("""
        INSERT INTO sensor_logs (timestamp, slot_id, occupied, distance_cm)
        VALUES (?, ?, ?, ?)
    """, (timestamp, slot_id, occupied, distance_cm))
    log_id = cursor.lastrowid

    # 2. Update parking_slots current status
    status_str = "occupied" if occupied == 1 else "available"
    cursor.execute("""
        UPDATE parking_slots
        SET status = ?, last_updated = ?
        WHERE slot_id = ?
    """, (status_str, timestamp, slot_id))

    conn.commit()
    conn.close()
    return log_id


def get_all_slots() -> List[Dict[str, Any]]:
    """Returns the current status of all parking slots."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT slot_id, location_label, status, last_updated
        FROM parking_slots
        ORDER BY slot_id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_slot_by_id(slot_id: str) -> Optional[Dict[str, Any]]:
    """Returns the current status for a specific parking slot ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT slot_id, location_label, status, last_updated
        FROM parking_slots
        WHERE slot_id = ?
    """, (slot_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_history_logs(slot_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Returns historical sensor logs, optionally filtered by slot_id.
    Ordered by id descending (most recent first).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    if slot_id:
        cursor.execute("""
            SELECT id, timestamp, slot_id, occupied, distance_cm
            FROM sensor_logs
            WHERE slot_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (slot_id, limit))
    else:
        cursor.execute("""
            SELECT id, timestamp, slot_id, occupied, distance_cm
            FROM sensor_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
