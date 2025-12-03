# MONAI + FastAPI Inference Deployment Tutorial

This tutorial demonstrates how to deploy MONAI model bundles as production-ready REST APIs using FastAPI.

## 📚 Overview

Learn how to:
- Load and serve MONAI model bundles
- Create FastAPI endpoints for medical image inference
- Handle medical image uploads (NIfTI format)
- Deploy with Docker for production
- Test and monitor your deployed model

## 🎯 What You'll Build

A complete REST API service that:
- ✅ Loads a pre-trained MONAI model (spleen CT segmentation)
- ✅ Accepts medical image uploads via HTTP
- ✅ Returns inference results in JSON format
- ✅ Includes auto-generated API documentation
- ✅ Runs in Docker containers for easy deployment

## 📋 Prerequisites

- Python 3.9+ installed
- Docker installed (for containerization)
- Basic knowledge of Python and REST APIs
- Familiarity with medical imaging (helpful but not required)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Locally

```bash
# From the fastapi_inference directory
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**View API Documentation:**
Open `http://localhost:8000/docs` in your browser

**Make a Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/image.nii.gz"
```

## 📁 Project Structure

```
fastapi_inference/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                  # FastAPI app and routes
│   ├── model_loader.py          # MONAI model loading (singleton)
│   ├── inference.py             # Inference logic
│   └── schemas.py               # Pydantic models for validation
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_api.py              # API endpoint tests
├── docker/                      # Docker configuration
│   ├── Dockerfile               # Container definition
│   └── docker-compose.yml       # Orchestration
├── notebooks/                   # Interactive tutorials
│   └── fastapi_tutorial.ipynb   # Step-by-step walkthrough
└── examples/                    # Usage examples
    ├── client.py                # Python client example
    └── sample_requests.http     # HTTP request examples
```

## 🔧 API Endpoints

### `GET /`
Returns API information

### `GET /health`
Health check endpoint
- Returns service status
- Indicates if model is loaded
- Shows computation device (CPU/GPU)

**Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### `POST /predict`
Run inference on uploaded medical image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (NIfTI format: .nii or .nii.gz)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "shape": [1, 2, 96, 96, 96],
    "min_value": 0.0,
    "max_value": 1.0,
    "unique_labels": [0, 1],
    "num_labels": 2
  },
  "segmentation_shape": [1, 2, 96, 96, 96],
  "metadata": {
    "image_shape": [1, 1, 96, 96, 96],
    "processing_time": 2.345,
    "device": "cuda"
  },
  "message": "Inference completed successfully in 2.345s"
}
```

### `GET /docs`
Interactive API documentation (Swagger UI)

### `GET /redoc`
Alternative API documentation (ReDoc)

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t monai-fastapi -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 monai-fastapi
```

### Or use Docker Compose

```bash
# Start the service
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop the service
docker-compose -f docker/docker-compose.yml down
```

## 📝 Usage Examples

### Python Client

```python
from examples.client import MONAIClient

# Initialize client
client = MONAIClient(base_url="http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Make prediction
result = client.predict("path/to/image.nii.gz")
print(result)
```

### Command Line

```bash
# Check health
python examples/client.py --health

# Run prediction
python examples/client.py --image path/to/image.nii.gz
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@tests/sample_image.nii.gz"
```

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 🔍 Model Information

**Default Model:** spleen_ct_segmentation

This tutorial uses MONAI's spleen CT segmentation bundle, which:
- Segments spleen from CT scans
- Pre-trained on Medical Segmentation Decathlon dataset
- Fast inference (~2-3 seconds on GPU)
- Good starting point for learning deployment

**To use a different model:**
Edit `app/main.py` and change the model name in the `lifespan` function:
```python
model_loader.load_model(
    model_name="your_model_name",  # Change this
    bundle_dir="./models"
)
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Server configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Model configuration
MODEL_NAME=spleen_ct_segmentation
MODEL_DIR=./models

# Performance
WORKERS=1
```

### GPU Support

The application automatically detects and uses GPU if available:
- **With GPU:** Faster inference, handles larger images
- **Without GPU:** Runs on CPU (slower but works)

## 🚦 Production Considerations

### Security
- Add authentication (JWT, API keys)
- Validate file sizes and types
- Use HTTPS in production
- Set CORS origins explicitly

### Performance
- Use multiple worker processes for scaling
- Add caching for frequently used models
- Implement request queuing for high load
- Consider model quantization for speed

### Monitoring
- Add logging and metrics
- Track inference times
- Monitor memory usage
- Set up health check endpoints

### Example Production Command

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --proxy-headers \
  --forwarded-allow-ips='*'
```

## 🐛 Troubleshooting

### Model Download Fails
```
Error: Failed to download model bundle
Solution: Check internet connection and MONAI bundle name
```

### Out of Memory
```
Error: CUDA out of memory
Solution: Reduce batch size or use CPU with smaller model
```

### File Format Error
```
Error: Invalid file format
Solution: Ensure file is NIfTI format (.nii or .nii.gz)
```

### Port Already in Use
```
Error: Address already in use
Solution: Change port or kill process using port 8000
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Model Zoo](https://monai.io/model-zoo.html)
- [MONAI Bundle Guide](https://docs.monai.io/en/stable/bundle_intro.html)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contributing

This tutorial is part of the MONAI tutorials collection. Contributions welcome!

## 📄 License

Copyright 2025 MONAI Consortium
Licensed under the Apache License, Version 2.0

## 🙋 Support

For questions about this tutorial:
- Open an issue on GitHub
- Visit MONAI community forums
- Check existing tutorials for similar examples

---

**Next Steps:**
1. ✅ Run through the tutorial
2. ✅ Experiment with different models
3. ✅ Deploy to your infrastructure
4. ✅ Build your own medical AI application!
