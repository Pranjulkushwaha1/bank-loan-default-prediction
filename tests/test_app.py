from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)   # ye fast api app ko test mode me chalata hai

def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}


def test_predict_valid():
    response = client.post("/v1/predict", json={
        "RevolvingUtilizationOfUnsecuredLines": 0,
        "age": 0,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0,
        "MonthlyIncome": 0,
        "NumberOfOpenCreditLinesAndLoans": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert "risk_category" in response.json()

def test_predict_invalid():
        response = client.post("/v1/predict", json={
             "age":"abs"
        })
        assert response.status_code == 422