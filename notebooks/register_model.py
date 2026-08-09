import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bank-loan-default-prediction")

model = joblib.load("models/best_model.pkl")

with mlflow.start_run(run_name="register-best-model"):
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LoanDefaultModel"
    )
    mlflow.log_metric("f1_score", 0.9468)

print("Model registered successfully!")