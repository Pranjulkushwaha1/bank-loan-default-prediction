import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("data/cs-training.csv")
print(df.shape)

engine = create_engine("mysql+pymysql://root:@localhost/bank_loan_db")

df.to_sql("loan_data", con=engine, if_exists="replace", index=False)
print("Data loaded successfully!")