from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)   # ye fast api app ko test mode me chalata hai

def get_auth_token():
    response = client.post("/auth/token", data={
        "username": "user",
        "password": "user123"
    })
    return response.json()["access_token"]

def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}


def test_predict_valid():
    token = get_auth_token()
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
    },
    headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert "risk_category" in response.json()

def test_predict_invalid():
    token = get_auth_token()          # ← ye add karo
    response = client.post("/v1/predict", json={
         "age":"abs"
    }, headers={"Authorization": f"Bearer {token}"})  # ← ye add karo
    assert response.status_code == 422