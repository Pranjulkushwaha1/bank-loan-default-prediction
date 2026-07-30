from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
app = FastAPI()

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

    
@app.post("/predict")
def predict(data: LoanApplication):
    
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
    
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk_category": risk
    }