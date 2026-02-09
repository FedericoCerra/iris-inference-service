# train.py - The "Production" Script
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load Data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. parameters found in exploration.ipynb (the Optuna notebook)

BEST_PARAMS = {
    "n_estimators": 44,
    "max_depth": 18,
    "min_samples_split": 17,
    "min_samples_leaf": 3,
    "max_features": "sqrt"
}

# 3. Build the Pipeline using those specific numbers
print(f"Training final model with params: {BEST_PARAMS}...")

model = RandomForestClassifier(
    **BEST_PARAMS,   # This unpacks the dictionary above
    random_state=42, 
    n_jobs=-1
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", model)
])

# 4. Train Once
pipeline.fit(X_train, y_train)

# 5. Verify & Save
score = pipeline.score(X_test, y_test)
print(f"Final Accuracy: {score:.4f}")

joblib.dump(pipeline, "models/iris_model_final.pkl")
