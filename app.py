from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Yield Prediction API")
model = joblib.load('crop_yield_model.pkl')

class PredictionInput(BaseModel):
    rainfall_mm: float
    avg_temp_c: float
    soil_ph: float
    nitrogen_ppm: float
    ndvi_index: float

@app.post("/predict")
def predict(data: PredictionInput):
    df_input = pd.DataFrame([data.dict()])
    prediction = model.predict(df_input)[0]
    return {"predicted_yield_kg_ha": round(float(prediction), 2)}