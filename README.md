# 🌸 Iris Inference Service

[![Live Demo](https://img.shields.io/badge/demo-live_green?style=for-the-badge&logo=render)](https://iris-inference-service.onrender.com/docs)
[![Docker Image Size (latest)](https://img.shields.io/docker/image-size/fededee/iris-inference/latest)](https://hub.docker.com/r/fededee/iris-inference)
![Python Version](https://img.shields.io/badge/python-3.11-blue)

A production-ready Machine Learning API that predicts Iris flower species. Built with **FastAPI**, **Scikit-Learn**, and **Docker**.

**Try the API live:** [https://iris-inference-service.onrender.com](https://iris-inference-service.onrender.com)
*(Note: The free server sleeps after 15 mins of inactivity. Please wait 30s for it to wake up!)*

## Features
- **Rest API:** Fast and validating API using FastAPI.
- **Containerized:** Docker image available on Docker Hub.
- **Optimized:** Hyperparameters tuned with Optuna.
- **Robust:** Includes probability scores in predictions.

##Installation & Usage

Option 1: Run with Docker (Recommended)
You don't need to install Python or clone this repo. Just run:
```bash
docker run -d -p 8000:80 fededee/iris-inference:latest
```

Then open your browser to: http://localhost:8000/docs

Option 2: Run Locally (For Developers)
If you want to modify the code:

Clone the repo:

```bash
 git clone [https://github.com/fededee/iris-inference-service.git](https://github.com/fededee/iris-inference-service.git)
 cd iris-inference-service
```

Install dependencies:
```bash
 pip install -r app/requirements.txt
```
Run the server:
```bash
uvicorn app.main:app --reload
```

## Model Information
Algorithm: Random Forest Classifier

Accuracy: ~98% on test set


## Expected input/output
Example Request:

```
JSON
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example Response:
```
JSON
{
  "class": "setosa",
  "class_id": 0,
  "probability": 0.98
}
```
