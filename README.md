# 🌸 Iris Inference Service

[![Docker Image Size (latest)](https://img.shields.io/docker/image-size/fededee/iris-inference/latest)](https://hub.docker.com/r/fededee/iris-inference)
![Python Version](https://img.shields.io/badge/python-3.11-blue)

A production-ready Machine Learning API that predicts Iris flower species. Built with **FastAPI**, **Scikit-Learn**, and **Docker**.

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

Input: Sepal Length, Sepal Width, Petal Length, Petal Width.

Output: Predicted Class (Setosa, Versicolor, Virginica) + Probability.
