#pip install uvicorn pydantic fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel): 
    YearsAtCompany:float
    EmployeeSatisfaction:float
    Position:str
    Salary:int

with open('rfmodel.pkl', 'rb') as f: 
    model = pickle.load(f)

@app.post('/')
async def score_model(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=list(item.dict().keys()))
    yhat = model.predict(df)
    return {"prediction":int(yhat)} 