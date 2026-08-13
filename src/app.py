from fastapi import FastAPI,Depends,HTTPException,status
from fastapi.security import OAuth2PasswordRequestForm
import joblib
import numpy as np
from pydantic import BaseModel

from src.auth import verify_password,create_access_token,get_current_user,require_role,fake_user_db

from datetime import datetime

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from loguru import logger
import sqlite3
logger.add(
    "logs/predictions.log",
    rotation="1 MB",
    enqueue=True
)
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        monthly_income REAL,
        probability REAL,
        risk_category TEXT,
        age INT
        )
""")
    conn.commit()
    conn.close()


model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')


app = FastAPI()
init_db()

def save_prediction(age, monthly_income, probability, risk_category):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (timestamp, age, monthly_income, probability, risk_category)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
          age, monthly_income, probability, risk_category))
    conn.commit()
    conn.close()

# Rate limiting setup — max 5 requests per minute per IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class LoanApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_user_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/v1/predict")
@limiter.limit("5/minute")
async def predict(request: Request, data: LoanApplication, current_user:dict=Depends(require_role("user"))):
    
    # Feature Engineering
    monthly_debt = data.MonthlyIncome * data.DebtRatio
    total_late = data.NumberOfTime30_59DaysPastDueNotWorse + data.NumberOfTime60_89DaysPastDueNotWorse + data.NumberOfTimes90DaysLate
    credit_risk = data.RevolvingUtilizationOfUnsecuredLines * data.NumberOfOpenCreditLinesAndLoans
    income_per_dependent = data.MonthlyIncome / (data.NumberOfDependents + 1)
    
    # Input array banao
    input_data = [[
        data.RevolvingUtilizationOfUnsecuredLines,
        data.age,
        data.NumberOfTime30_59DaysPastDueNotWorse,
        data.DebtRatio,
        data.MonthlyIncome,
        data.NumberOfOpenCreditLinesAndLoans,
        data.NumberOfTimes90DaysLate,
        data.NumberRealEstateLoansOrLines,
        data.NumberOfTime60_89DaysPastDueNotWorse,
        data.NumberOfDependents,
        monthly_debt,
        total_late,
        credit_risk,
        income_per_dependent
    ]]
    
    # Scale karo
    input_scaled = scaler.transform(input_data)
    
    # Predict karo
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Risk category
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    logger.info(f"Prediction: {risk} | Probability: {probability}")
    save_prediction(data.age, data.MonthlyIncome, probability, risk)

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk_category": risk
    }
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy"}