import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine("postgresql+psycopg2://postgres:154269@localhost/bank_loan_db")

df = pd.read_csv("data/cs-training.csv")
print(df.shape)

df.to_sql("loan_data", con=engine, if_exists="replace", index=False)
print("Data loaded successfully!")