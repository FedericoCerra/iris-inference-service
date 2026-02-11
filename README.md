# 🌸 Iris Inference Service

[![Live Demo](https://img.shields.io/badge/demo-live_green?style=for-the-badge&logo=render)](https://iris-inference-service.onrender.com/docs)
[![Docker Image Size (latest)](https://img.shields.io/docker/image-size/fededee/iris-inference/latest)](https://hub.docker.com/r/fededee/iris-inference)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit)](https://iris-inference-service.streamlit.app)

A Full Stack Machine Learning Application that predicts Iris flower species. It features a robust FastAPI backend for inference and an interactive Streamlit frontend for data exploration.

**Try the data visualizer:** [https://iris-inference-service.streamlit.app](https://iris-inference-service.streamlit.app)

**Try the API live:** [https://iris-inference-service.onrender.com](https://iris-inference-service.onrender.com)
*(Note: The free server sleeps after 15 mins of inactivity. Please wait 30s for it to wake up!)*

## Features
### Frontend (Streamlit)

- **Interactive 3D Analysis**  
  Explore the Iris dataset in 3D feature space and visually inspect how species cluster.

- **Smart Radar Comparison**  
  Overlay your input flower’s feature profile against the *average* profiles of all three species.

- **Real-Time Metrics**  
  Instant feedback showing how your input differs from the norm  
  (e.g. *“Sepal Length is +1.2 cm above average”*).

---

### Backend (FastAPI)

- **REST API**  
  Fast, type-safe, and fully validated API built with FastAPI.

- **Optimized Model**  
  Random Forest Classifier tuned with Optuna  
  (~98% accuracy on the test set).

- **Containerized**  
  Docker image published and ready to run.

All good — here’s that section converted into clean, copy-pasteable GitHub Markdown.
You can drop this straight into your README.md.

## 🚀 Usage (No Installation Required)

You can try the full application directly in your browser — no setup needed.

### 🌸 Frontend App
https://iris-inference-service.streamlit.app

Interactive dashboard to visualize the Iris dataset and make real-time predictions.

### ⚡ Backend API
https://iris-inference-service.onrender.com/docs

Live Swagger UI to explore and test the API endpoints.

> ⚠️ Note: The free server sleeps after 15 minutes of inactivity.  
> If the app shows an error, wait ~30 seconds for the API to wake up.

---

## 🛠️ Local Installation (For Developers)

If you want to run the project locally or contribute, follow one of the options below.

---

### Option 1: Run Backend with Docker (Fastest)

No Python installation or repository cloning required.  
This pulls the production-ready Docker image directly.

```bash
docker run -d -p 8000:80 fededee/iris-inference:latest
```

The API will be available at:
http://localhost:8000/docs

Option 2: Run Full Stack from Source

Clone the repository to run both the API and the frontend locally.

1️. Clone the repository
```bash
git clone https://github.com/fededee/iris-inference-service.git
cd iris-inference-service
```
2️. Install dependencies
```bash
pip install -r requirements.txt
```
3️. Run the Backend (Terminal 1)
```bash
uvicorn main:app --reload
```

API runs at:
http://127.0.0.1:8000

4️. Run the Frontend (Terminal 2)
```bash
streamlit run streamlit_app.py
```

UI opens at:
http://localhost:8000

Frontend Configuration

By default, the Streamlit app uses the live production API.

To use your local backend instead:

Open streamlit_app.py

Comment out the production URL and uncomment the localhost URL:
```
# API_URL = "https://iris-inference-service.onrender.com/predict"  # Production API
API_URL = "http://127.0.0.1:8000/predict"
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
