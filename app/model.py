import joblib
import os

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

class IrisModel:
    def __init__(self):
        self.model = None

    def load_model(self, model_path="../models/iris_model_final.pkl"):
        model_path = os.path.join(os.path.dirname(__file__), model_path) 
        self.model = joblib.load(model_path)
        
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        prediction = self.model.predict([input_data])
        predicted_class_id = int(prediction[0])
        probability = max(self.model.predict_proba([input_data])[0])
        return predicted_class_id, SPECIES_MAP[predicted_class_id], probability
    
    

ai_model = IrisModel()
        