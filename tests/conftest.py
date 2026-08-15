# tests/conftest.py
import pytest
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

@pytest.fixture(autouse=True, scope="session")
def create_dummy_models():
    os.makedirs("models", exist_ok=True)
    X = np.random.rand(10, 14)
    y = np.array([0,1]*5)
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "models/best_model.pkl")
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, "models/scaler.pkl")