#pip install uvicorn pydantic fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# {
# "YearsAtCompany":1,
# "EmployeeSatisfaction":0.01,
# "Position":"Non-Manager",
# "Salary":4.0
# }

app = FastAPI()

class ScoringItem(BaseModel): 
    YearsAtCompany:float
    EmployeeSatisfaction:float
    Position:str
    Salary:int

with open('rfmodel.pkl', 'rb') as f: 
    model = pickle.load(f)

@app.get('/')
async def score_model(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=list(item.dict().keys()))
    yhat = model.predict(df)
    return {"prediction":int(yhat)} 