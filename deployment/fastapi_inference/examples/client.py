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
Example Python Client for MONAI FastAPI Inference Service

This script demonstrates how to interact with the deployed MONAI inference API.
"""

import argparse
import json
from pathlib import Path

import requests


class MONAIClient:
    """Client for interacting with MONAI FastAPI inference service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict:
        """
        Check if the service is healthy.

        Returns:
            Health status dictionary
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict(self, image_path: str) -> dict:
        """
        Send an image for inference.

        Args:
            image_path: Path to the medical image file (.nii or .nii.gz)

        Returns:
            Prediction results dictionary

        Raises:
            FileNotFoundError: If image file doesn't exist
            requests.HTTPError: If API request fails
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "application/octet-stream")}
            response = requests.post(f"{self.base_url}/predict", files=files)

        response.raise_for_status()
        return response.json()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="MONAI FastAPI Inference Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--health", action="store_true", help="Check API health status")
    parser.add_argument("--image", type=str, help="Path to medical image file for prediction")

    args = parser.parse_args()

    # Initialize client
    client = MONAIClient(base_url=args.url)

    # Health check
    if args.health:
        print("Checking API health...")
        try:
            health = client.health_check()
            print(json.dumps(health, indent=2))
        except requests.RequestException as e:
            print(f"Error: {e}")
            return 1

    # Prediction
    if args.image:
        print(f"Sending image for prediction: {args.image}")
        try:
            result = client.predict(args.image)
            print("\nPrediction Results:")
            print(json.dumps(result, indent=2))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except requests.HTTPError as e:
            print(f"API Error: {e}")
            print(f"Response: {e.response.text}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
