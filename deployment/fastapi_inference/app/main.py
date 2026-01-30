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
FastAPI Application for MONAI Model Inference

This module provides a REST API for deploying MONAI model bundles.
It demonstrates how to serve medical imaging AI models in production.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .inference import inference_engine
from .model_loader import model_loader
from .schemas import HealthResponse, PredictionResponse, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup: Load the MONAI model
    logger.info("Starting up: Loading MONAI model...")
    try:
        model_loader.load_model(model_name="spleen_ct_segmentation", bundle_dir="./models")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue anyway - model loading can be retried

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="MONAI Inference API",
    description="REST API for deploying MONAI model bundles for medical image inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500,
        },
    )


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns basic API information",
)
async def root():
    """Root endpoint - API information."""
    return {
        "name": "MONAI Inference API",
        "version": "1.0.0",
        "description": "FastAPI deployment for MONAI models",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service and model are ready",
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: Service and model status
    """
    is_loaded = model_loader.is_loaded()

    return HealthResponse(
        status="healthy" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
        device=str(model_loader.device) if is_loaded else "unknown",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run inference",
    description="Upload a medical image and get predictions",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def predict(file: UploadFile = File(..., description="Medical image file (NIfTI format: .nii or .nii.gz)")):
    """
    Run inference on uploaded medical image.

    Args:
        file: Uploaded image file (NIfTI format)

    Returns:
        PredictionResponse: Prediction results with metadata

    Raises:
        HTTPException: If file format is invalid or inference fails
    """
    # Validate file format
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file format. Supported formats: .nii, .nii.gz"
        )

    # Check if model is loaded
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Please try again later."
        )

    try:
        # Read file content
        contents = await file.read()

        # Run inference
        result = await inference_engine.process_image(image_bytes=contents, filename=file.filename)

        return PredictionResponse(**result)

    except ValueError as e:
        # Client error (bad input)
        logger.warning(f"Bad request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Server error (inference failed)
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during prediction"
        )


if __name__ == "__main__":
    import uvicorn

    # For development only - use proper ASGI server in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
