import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

df = pd.read_csv("data/dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model trained and saved!")

print("Training script ran successfully")
