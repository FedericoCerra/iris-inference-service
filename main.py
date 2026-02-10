from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.schemas import IrisInput, IrisPrediction
from app.model import ai_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_model.load_model()
    yield
    
app = FastAPI(title="Iris Recognition API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Recognition API! Use /predict to get predictions."}

@app.post("/predict", response_model=IrisPrediction)
def predict(input_data: IrisInput):
    input_list = [
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]
    class_id, species = ai_model.predict(input_list)
    return {"class_id": class_id, "species": species}