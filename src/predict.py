import dill
import pandas as pd
import numpy as np

def load_model(model_path="models/model_xgb.pkl"):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def predict(input_data: dict, model_path="models/model_xgb.pkl"):
    model = load_model(model_path)

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
