# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    prediction: Optional[Dict] = Field(None, description="Prediction results (format depends on model output)")
    segmentation_shape: Optional[List[int]] = Field(None, description="Shape of segmentation mask if applicable")
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    message: Optional[str] = Field(None, description="Additional information or error message")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
