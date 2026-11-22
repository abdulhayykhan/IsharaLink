"""
schemas.py -- Pydantic request and response models for IsharaLink backend.

Enforces the fixed interface contract agreed upon across frontend, backend,
and data leads.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class HandPoint(BaseModel):
    """A single 3D landmark [x, y, z]."""
    x: float
    y: float
    z: float


class Frame(BaseModel):
    """
    One captured video frame containing left and right hand landmarks.
    Each hand is either None/null (missing) or a list of 21 [x, y, z] points.
    """
    left: Optional[List[List[float]]] = Field(
        default=None,
        description="21 landmarks [x, y, z] for left hand, or null if hand absent"
    )
    right: Optional[List[List[float]]] = Field(
        default=None,
        description="21 landmarks [x, y, z] for right hand, or null if hand absent"
    )

    @field_validator("left", "right")
    @classmethod
    def validate_hand_points(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Hand landmarks must be a list or null")
        if len(v) != 21:
            raise ValueError(f"Hand must have exactly 21 landmark points, got {len(v)}")
        for i, pt in enumerate(v):
            if not isinstance(pt, (list, tuple)) or len(pt) != 3:
                raise ValueError(f"Point {i} must be [x, y, z] coordinates of length 3")
            for coord in pt:
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"Coordinate values must be numeric, got {coord}")
        return v


class PredictRequest(BaseModel):
    """
    Inference request from client.
    Must contain a non-empty sequence of frames (minimum 10 frames).
    """
    frames: List[Frame] = Field(
        ...,
        description="Sequence of raw frames captured from camera (~1-2 seconds, min 10 frames)"
    )
    client: Optional[str] = Field(default="web", description="Client identifier (e.g. web, mobile)")

    @field_validator("frames")
    @classmethod
    def validate_frame_count(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Clip cannot be empty; 'frames' list is required")
        if len(v) < 10:
            raise ValueError(f"Clip too short: received {len(v)} frames, minimum required is 10 frames")
        return v


class Top3Item(BaseModel):
    label: str
    p: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top3: List[Top3Item]
    status: str = Field(description="'ok' if confidence >= threshold else 'unknown'")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes_count: int
    threshold: float


class ClassesResponse(BaseModel):
    classes: List[str]
    total: int
