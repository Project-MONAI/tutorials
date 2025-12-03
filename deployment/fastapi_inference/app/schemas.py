"""
Pydantic Models for Request/Response Validation

This module defines the data structures for API requests and responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Computation device (CPU/GPU)")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    image_shape: List[int] = Field(..., description="Input image dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")
    device: str = Field(..., description="Device used for inference")


class PredictionResponse(BaseModel):
    """Response model for inference predictions."""

    success: bool = Field(..., description="Whether prediction was successful")
    prediction: Optional[Dict] = Field(
        None,
        description="Prediction results (format depends on model output)"
    )
    segmentation_shape: Optional[List[int]] = Field(
        None,
        description="Shape of segmentation mask if applicable"
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    message: Optional[str] = Field(None, description="Additional information or error message")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
