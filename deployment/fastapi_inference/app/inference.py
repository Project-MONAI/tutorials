"""
Inference Logic

This module handles the preprocessing, inference, and postprocessing
of medical images using MONAI models.
"""

import logging
import time
from io import BytesIO
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    ScaleIntensity,
    EnsureType,
)

from .model_loader import model_loader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles image preprocessing, inference, and postprocessing."""

    def __init__(self):
        """Initialize the inference engine with preprocessing transforms."""
        self.preprocess = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Spacing(pixdim=(1.5, 1.5, 2.0)),
                ScaleIntensity(),
                EnsureType(dtype=torch.float32),
            ]
        )

    async def process_image(self, image_bytes: bytes, filename: str) -> Dict:
        """
        Process an uploaded image and return predictions.

        Args:
            image_bytes: Raw bytes of the uploaded image
            filename: Original filename (for logging)

        Returns:
            Dictionary containing prediction results and metadata

        Raises:
            ValueError: If image format is unsupported
            RuntimeError: If inference fails
        """
        start_time = time.time()

        try:
            # Save bytes to temporary file-like object
            image_buffer = BytesIO(image_bytes)

            # Load and preprocess image
            logger.info(f"Processing image: {filename}")
            image_data = self._load_image(image_buffer, filename)
            image_tensor = self._preprocess_image(image_data)

            # Run inference
            prediction = await self._run_inference(image_tensor)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare response
            result = {
                "success": True,
                "prediction": self._format_prediction(prediction),
                "segmentation_shape": (
                    list(prediction.shape) if isinstance(prediction, (np.ndarray, torch.Tensor)) else None
                ),
                "metadata": {
                    "image_shape": list(image_tensor.shape),
                    "processing_time": round(processing_time, 3),
                    "device": str(model_loader.device),
                },
                "message": f"Inference completed successfully in {processing_time:.3f}s",
            }

            logger.info(f"Inference completed in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference error: {str(e)}")

    def _load_image(self, image_buffer: BytesIO, filename: str) -> np.ndarray:
        """
        Load image from bytes buffer.

        Args:
            image_buffer: BytesIO object containing image data
            filename: Original filename for format detection

        Returns:
            Loaded image as numpy array

        Raises:
            ValueError: If image format is unsupported
        """
        try:
            # Support NIfTI format (.nii, .nii.gz)
            if filename.endswith((".nii", ".nii.gz")):
                image_buffer.seek(0)
                img = nib.load(image_buffer)
                return img.get_fdata()
            else:
                raise ValueError(f"Unsupported image format: {filename}. " "Supported formats: .nii, .nii.gz")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def _preprocess_image(self, image_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_data: Raw image data as numpy array

        Returns:
            Preprocessed image tensor
        """
        try:
            # Add batch dimension if needed
            image_tensor = torch.from_numpy(image_data)

            # Ensure batch dimension
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add channel
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch

            # Move to device
            image_tensor = image_tensor.to(model_loader.device)

            return image_tensor

        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {str(e)}")

    async def _run_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run model inference.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Model prediction

        Raises:
            RuntimeError: If inference fails
        """
        try:
            model = model_loader.model

            # Run inference with no gradient computation
            with torch.no_grad():
                if hasattr(model, "__call__"):
                    prediction = model(image_tensor)
                else:
                    raise RuntimeError("Model is not callable")

            return prediction

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")

    def _format_prediction(self, prediction: torch.Tensor) -> Dict:
        """
        Format prediction output for JSON response.

        Args:
            prediction: Raw model output

        Returns:
            Formatted prediction dictionary
        """
        try:
            # Convert to numpy
            if isinstance(prediction, torch.Tensor):
                pred_np = prediction.cpu().numpy()
            else:
                pred_np = prediction

            # Basic statistics
            result = {
                "shape": list(pred_np.shape),
                "dtype": str(pred_np.dtype),
                "min_value": float(pred_np.min()),
                "max_value": float(pred_np.max()),
                "mean_value": float(pred_np.mean()),
            }

            # For segmentation masks, add unique labels
            if pred_np.ndim >= 3:
                unique_labels = np.unique(pred_np.astype(int))
                result["unique_labels"] = unique_labels.tolist()
                result["num_labels"] = len(unique_labels)

            return result

        except Exception as e:
            logger.warning(f"Failed to format prediction: {str(e)}")
            return {"raw_type": str(type(prediction))}


# Global inference engine instance
inference_engine = InferenceEngine()
