import os
import pandas as pd
import joblib
import pytest
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"

@pytest.fixture(scope="module")
def load_dataset():
    """Load dataset if it exists, skip tests otherwise."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Dataset not found. Make sure DVC pull ran before tests.")
    df = pd.read_csv(DATA_PATH, sep="\t")
    return df

def test_data_loading(load_dataset):
    df = load_dataset
    # Check dataframe is not empty and has label column
    assert not df.empty
    assert "label" in df.columns

def test_model_training_and_save(load_dataset):
    df = load_dataset
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Check model is saved correctly
    assert os.path.exists(MODEL_PATH)
    loaded_model = joblib.load(MODEL_PATH)
    assert isinstance(loaded_model, LogisticRegression)

def test_shape_validation(load_dataset):
    df = load_dataset
    X = df.drop("label", axis=1)
    y = df["label"]

    # Features and labels have same number of rows
    assert X.shape[0] == y.shape[0]

    # Optional: check number of features
    expected_features = X.shape[1]
    assert expected_features > 0
