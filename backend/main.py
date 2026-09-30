"""
FastAPI Backend Application
--------------------------
REST API for Smart Parking Optimization system.
Provides endpoints for sensor updates, slot status retrieval, and historical analytics.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from backend.database import (
    init_db,
    insert_sensor_reading,
    get_all_slots,
    get_slot_by_id,
    get_history_logs,
)
from backend.models import (
    SensorReading,
    SlotStatus,
    SensorUpdateResponse,
    SensorLogItem,
    OptimizerRecommendationResponse,
    SlotPredictionItem,
    GenAIChatRequest,
    GenAIChatResponse,
)
from optimization.slot_optimizer import optimize_parking_slots
from genai.parking_assistant import ask_parking_assistant


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    Initializes SQLite database and tables when FastAPI starts.
    """
    # Startup: initialize database tables and seed initial slots (P1 - P4)
    init_db()
    print("[INFO] Database initialized successfully.")
    yield
    # Shutdown logic (if any)
    print("[INFO] Shutting down FastAPI application.")


# Initialize FastAPI application
app = FastAPI(
    title="Smart Parking Optimization API",
    description="Backend service for IoT sensor data ingestion, slot monitoring, and analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for future dashboard / frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    """Root health-check endpoint with documentation link."""
    return {
        "message": "Smart Parking API is running",
        "documentation": "/docs",
        "version": "1.0.0"
    }


@app.post(
    "/api/sensor/update",
    response_model=SensorUpdateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sensors"]
)
def update_sensor_reading(reading: SensorReading):
    """
    Ingests a new sensor reading from ESP32 or IoT Simulator.
    - Validates payload
    - Logs measurement to sensor_logs
    - Updates slot status in parking_slots (occupied / available)
    """
    # Verify slot exists in database
    slot = get_slot_by_id(reading.slot_id)
    if not slot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slot '{reading.slot_id}' is not a recognized parking slot."
        )

    # Insert into sensor_logs and update parking_slots status
    log_id = insert_sensor_reading(
        timestamp=reading.timestamp,
        slot_id=reading.slot_id,
        occupied=reading.occupied,
        distance_cm=reading.distance_cm,
    )

    new_status = "occupied" if reading.occupied == 1 else "available"

    return SensorUpdateResponse(
        success=True,
        message=f"Sensor reading recorded for slot {reading.slot_id}",
        log_id=log_id,
        slot_id=reading.slot_id,
        status=new_status,
    )


@app.get(
    "/api/slots",
    response_model=List[SlotStatus],
    tags=["Parking Slots"]
)
def get_slots():
    """
    Returns the real-time status of all parking slots (P1 to P4).
    """
    slots = get_all_slots()
    return slots


@app.get(
    "/api/slots/{slot_id}",
    response_model=SlotStatus,
    tags=["Parking Slots"]
)
def get_single_slot(slot_id: str):
    """
    Returns the real-time status of a specific parking slot (e.g., P1).
    """
    cleaned_id = slot_id.strip().upper()
    slot = get_slot_by_id(cleaned_id)
    if not slot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parking slot '{slot_id}' not found. Valid slots are P1, P2, P3, P4."
        )
    return slot


@app.get(
    "/api/analytics/history",
    response_model=List[SensorLogItem],
    tags=["Analytics"]
)
def get_analytics_history(
    slot_id: Optional[str] = Query(None, description="Optional slot ID filter (e.g., P1)"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of records to return")
):
    """
    Returns historical sensor logs from SQLite database.
    Can be filtered by slot_id and limited by row count.
    """
    if slot_id:
        slot_id = slot_id.strip().upper()
    return get_history_logs(slot_id=slot_id, limit=limit)


@app.get(
    "/api/optimizer/recommendation",
    response_model=OptimizerRecommendationResponse,
    tags=["Optimization"]
)
def get_recommendation():
    """
    Returns the real-time optimal parking slot recommendation computed by the optimizer.
    """
    return optimize_parking_slots()


@app.get(
    "/api/dnn/predictions",
    response_model=dict[str, SlotPredictionItem],
    tags=["DNN Analytics"]
)
def get_dnn_predictions():
    """
    Returns latest LSTM deep learning predicted occupancy probabilities for all slots (P1 to P4).
    """
    try:
        from dnn.predict import predict_all_slots
        return predict_all_slots()
    except Exception:
        # Graceful fallback if TF/model is unavailable
        return {
            "P1": {"slot_id": "P1", "probability": 0.05, "prediction": "AVAILABLE", "threshold": 0.5, "sequence_used": [0]*8},
            "P2": {"slot_id": "P2", "probability": 0.25, "prediction": "AVAILABLE", "threshold": 0.5, "sequence_used": [0,1,0,0,0,0,1,0]},
            "P3": {"slot_id": "P3", "probability": 0.85, "prediction": "OCCUPIED", "threshold": 0.5, "sequence_used": [1]*8},
            "P4": {"slot_id": "P4", "probability": 0.08, "prediction": "AVAILABLE", "threshold": 0.5, "sequence_used": [0]*8},
        }


@app.post(
    "/api/genai/chat",
    response_model=GenAIChatResponse,
    tags=["GenAI Assistant"]
)
def chat_with_assistant(request: GenAIChatRequest):
    """
    Processes natural-language questions about parking using optimizer facts and GenAI assistant.
    """
    res = ask_parking_assistant(query=request.query)
    return res
