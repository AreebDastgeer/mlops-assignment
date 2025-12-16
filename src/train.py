import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

dataset_path = "data/dataset.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv("data/dataset.csv", sep="\t")  # tab-separated
    print("Dataset loaded")

    # Prepare features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved!")

else:
    print("Dataset not found, skipping training for CI")
