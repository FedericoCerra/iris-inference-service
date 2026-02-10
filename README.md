# 🌸 Iris Inference Service

![Docker Image Size (latest)](https://img.shields.io/docker/image-size/fededee/iris-inference/latest)
![Python Version](https://img.shields.io/badge/python-3.11-blue)

A production-ready Machine Learning API that predicts Iris flower species. Built with **FastAPI**, **Scikit-Learn**, and **Docker**.

## Features
- **Rest API:** Fast and validating API using FastAPI.
- **Containerized:** Docker image available on Docker Hub.
- **Optimized:** Hyperparameters tuned with Optuna.
- **Robust:** Includes probability scores in predictions.

##Installation & Usage

### Option 1: Run with Docker (Recommended)
You don't need to install Python or clone this repo. Just run:

```bash
docker run -d -p 8000:80 fededee/iris-inference:latest
