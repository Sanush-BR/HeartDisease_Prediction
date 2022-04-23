import numpy 
import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Global model
with open('heart_model.pickle','rb') as f:
    __model = pickle.load(f)

app  = FastAPI()

class Heart(BaseModel):
    Age: int
    Sex: bool
    Cp:int
    Trestbps:int
    Chol:int
    Fbs:bool
    Restecg:int
    Thalach:int
    Exang:bool
    Oldpeak:float
    Slope:int
    Ca:int
    Thal:int

@app.get('/')
async def home():
    return "Welcome"

@app.post('/predict')
async def model(data:Heart):
    data = data.dict()

    age = data['Age']
    sex = data['Sex']
    cp = data['Cp']
    trestbps = data['Trestbps']
    chol = data['Chol']
    fbs = data['Fbs']
    restecg = data['Restecg']
    thalach = data['Thalach']
    exang = data['Exang']
    oldpeak = data['Oldpeak']
    slope = data['Slope']
    ca = data['Ca']
    thal = data['Thal']

    result = __model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])[0]
    
    if(result == 1):
        return "Pearson has Heart disease"
    return "Pearson is healthy"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
