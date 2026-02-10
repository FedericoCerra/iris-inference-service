from pydantic import BaseModel

# This class defines the Input structure
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    # This is for documentation in the API (Swagger UI)
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class IrisPrediction(BaseModel):
    species: str
    class_id: int
    probability: float