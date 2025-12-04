"""
MONAI Model Loader

This module implements a singleton pattern for loading and caching MONAI model bundles.
The model is loaded once at startup and reused for all inference requests.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from monai.bundle import download, load

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class for loading and managing MONAI model bundles.

    This ensures the model is loaded only once and reused across requests,
    improving performance and resource utilization.
    """

    _instance: Optional["ModelLoader"] = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model loader (called only once)."""
        if self._model is None:
            self._setup_device()

    def _setup_device(self):
        """Determine and set up the computation device (CPU or GPU)."""
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU for inference")

    def load_model(self, model_name: str = "spleen_ct_segmentation", bundle_dir: str = "./models") -> None:
        """
        Load a MONAI model bundle.

        Args:
            model_name: Name of the MONAI bundle to load
            bundle_dir: Directory to store/load the bundle

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            bundle_path = Path(bundle_dir) / model_name

            # Download bundle if not exists
            if not bundle_path.exists():
                logger.info(f"Downloading model bundle: {model_name}")
                download(name=model_name, bundle_dir=bundle_dir)
                logger.info(f"Model downloaded successfully to {bundle_path}")
            else:
                logger.info(f"Using existing model bundle at {bundle_path}")

            # Load the model
            logger.info("Loading model into memory...")
            self._model = load(name=model_name, bundle_dir=bundle_dir, source="monaihosting")

            # Move model to device
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)

            # Set model to evaluation mode
            if hasattr(self._model, "eval"):
                self._model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    @property
    def model(self):
        """Get the loaded model instance."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def device(self):
        """Get the computation device."""
        return self._device

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Global instance
model_loader = ModelLoader()
