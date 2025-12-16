import pickle
import joblib
import os

# Path to your old model (and where it will be re-saved)
model_path = "models/model.pkl"

# Try loading with pickle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded with pickle.")
except Exception as e:
    print("Pickle failed, trying joblib...")
    model = joblib.load(model_path)
    print("Model loaded with joblib.")

# Re-save the model using joblib in the same path
joblib.dump(model, model_path)
print(f"Model re-saved successfully at {model_path}")
