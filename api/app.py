# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import dill
import numpy as np
import traceback

# -----------------------------
# Load trained pipeline once
# -----------------------------
with open("models/model_xgb.pkl", "rb") as f:
    model = dill.load(f)

# -----------------------------
# Column mapping from API -> trained model
# -----------------------------
COLUMN_MAPPING = {
    "Age": "Age",
    "Sex": "Sex",
    "Chest_pain_type": "Chest pain type",
    "BP": "BP",
    "Cholesterol": "Cholesterol",
    "FBS_over_120": "FBS over 120",
    "EKG_results": "EKG results",
    "Max_HR": "Max HR",
    "Exercise_angina": "Exercise angina",
    "ST_depression": "ST depression",
    "Slope_of_ST": "Slope of ST",
    "Number_of_vessels_fluro": "Number of vessels fluro",
    "Thallium": "Thallium"
}

# -----------------------------
# Pydantic model for input
# -----------------------------
class HeartData(BaseModel):
    Age: float
    Sex: float
    Chest_pain_type: float
    BP: float
    Cholesterol: float
    FBS_over_120: float
    EKG_results: float
    Max_HR: float
    Exercise_angina: float
    ST_depression: float
    Slope_of_ST: float
    Number_of_vessels_fluro: float
    Thallium: float

# -----------------------------
# Helper: prepare input DataFrame
# -----------------------------
def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = df.rename(columns=COLUMN_MAPPING)
    df = df.astype(float)

    # derived feature (correct column name)
    df["Risk indicator"] = np.where(
        (df["Thallium"] > 6) & (df["Max HR"] < 150),
        1,
        0
    )

    return df

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Heart Disease Predictor")

@app.post("/predict")
def predict(heart_data: HeartData):
    try:
        df = prepare_input(heart_data.dict())
        pred = model.predict(df)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }