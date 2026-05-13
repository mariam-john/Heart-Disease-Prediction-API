import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import dill
from sklearn.model_selection import train_test_split


# -----------------------------
# preprocessing
# -----------------------------
def preprocessing(df):
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")

    df["Risk indicator"] = np.where(
        (df["Thallium"] > 6) & (df["Max HR"] < 150),
        1, 0
    )

    df["Cholesterol"] = np.log1p(df["Cholesterol"])
    df["ST depression"] = np.log1p(df["ST depression"])

    return df


# -----------------------------
# training
# -----------------------------
def train_model(data_path, model_path="models/model_xgb.pkl"):

    df = pd.read_csv(data_path)

    df["target_numeric"] = df["Heart Disease"].map({
        "Presence": 1,
        "Absence": 0
    })

    X = df.drop(["target_numeric", "Heart Disease"], axis=1)
    y = df["target_numeric"]

    # APPLY preprocessing BEFORE training
    X = preprocessing(X)

    columns = ["BP", "Max HR", "Age", "Cholesterol"]

    scaler = ColumnTransformer(
        [("num", StandardScaler(), columns)],
        remainder="passthrough"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=16,
        eval_metric="auc"
    )

    pipeline = Pipeline([
        ("scale", scaler),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        dill.dump(pipeline, f)

    print(f"✅ Model saved at {model_path}")


if __name__ == "__main__":
    train_model("data/raw/train.csv")