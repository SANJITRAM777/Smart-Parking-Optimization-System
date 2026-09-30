"""
Pydantic Models for Data Validation
----------------------------------
Defines request and response schemas for FastAPI endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """
    Schema for incoming sensor telemetry payload from ESP32 or IoT Simulator.
    """
    timestamp: str = Field(
        ...,
        description="Reading timestamp in ISO or 'YYYY-MM-DD HH:MM:SS' format",
        examples=["2026-09-26 10:00:00"]
    )
    slot_id: str = Field(
        ...,
        description="Slot identifier, e.g., P1, P2, P3, P4",
        examples=["P1"]
    )
    occupied: int = Field(
        ...,
        description="Binary occupancy status (1 = occupied, 0 = available)",
        ge=0,
        le=1,
        examples=[1]
    )
    distance_cm: float = Field(
        ...,
        description="Ultrasonic measured distance in centimeters (> 0)",
        gt=0,
        examples=[12.5]
    )

    @field_validator("slot_id")
    @classmethod
    def validate_slot_id(cls, v: str) -> str:
        cleaned = v.strip().upper()
        valid_slots = ["P1", "P2", "P3", "P4"]
        if cleaned not in valid_slots:
            raise ValueError(f"Invalid slot_id '{v}'. Allowed slots are: {', '.join(valid_slots)}")
        return cleaned


class SlotStatus(BaseModel):
    """
    Schema for current parking slot status.
    """
    slot_id: str
    location_label: str
    status: Literal["available", "occupied"]
    last_updated: Optional[str] = None


class SensorUpdateResponse(BaseModel):
    """
    Response schema after successfully recording a sensor reading.
    """
    success: bool
    message: str
    log_id: int
    slot_id: str
    status: str


class SensorLogItem(BaseModel):
    """
    Schema for a historical sensor reading record.
    """
    id: int
    timestamp: str
    slot_id: str
    occupied: int
    distance_cm: float


class OptimizerRankedSlot(BaseModel):
    """
    Schema for individual slot ranking in optimizer results.
    """
    slot_id: str
    status: str
    distance_m: float
    predicted_occupancy: float
    score: float


class OptimizerRecommendationResponse(BaseModel):
    """
    Schema for parking slot optimizer recommendation output.
    """
    recommended_slot: Optional[str] = None
    score: float
    reason: str
    distance_m: Optional[float] = None
    predicted_occupancy: Optional[float] = None
    ranked_slots: list[OptimizerRankedSlot] = []
    all_slots: list[OptimizerRankedSlot] = []


class SlotPredictionItem(BaseModel):
    """
    Schema for single slot DNN LSTM prediction result.
    """
    slot_id: str
    probability: float
    prediction: str
    threshold: float
    sequence_used: list[int]


class GenAIChatRequest(BaseModel):
    """
    Schema for user query to GenAI Parking Assistant.
    """
    query: str = Field(..., min_length=1, description="Natural language question about parking")


class GenAIChatResponse(BaseModel):
    """
    Schema for GenAI Parking Assistant natural language response.
    """
    query: str
    response: str
    mode: str
    recommended_slot: Optional[str] = None
    optimizer_score: Optional[float] = None
