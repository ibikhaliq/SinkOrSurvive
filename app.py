from fastapi import FastAPI
import pydantic
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from enum import Enum


app = FastAPI()

model = joblib.load("titanic_pipeline.joblib")

class SexEnum(str, Enum):
    male ="male"
    female ="female"

class EmbarkedEnum(str, Enum):
    C="C"
    Q="Q"
    S="S"

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, 3)")
    Sex: SexEnum
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0, le=8, description="Siblings/Spouses aboard")
    Parch: int = Field(..., ge=0, le=6, description="Parents/Children aboard")
    Fare: float = Field(..., ge=0)
    Embarked: EmbarkedEnum
    HasCabin: int = Field(..., ge=0, le=1)

class Prediction(BaseModel):
    prediction: int
    survival_probability: float

@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predict", response_model= Prediction)
def predict(passenger: Passenger):
    data = passenger.dict()
    df = pd.DataFrame([data])


    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["Family_Size"] == 1).astype(int)


    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    return {
        "prediction": int(pred),
        "survival_probability": round(float(proba), 3)
    }

@app.post("/predict-batch", response_model= list[Prediction])
def predict_batch(items: list[Passenger]):
    df = pd.DataFrame([x.dict() for x in items])
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["Family_Size"] == 1).astype(int)
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    return [{"prediction": int(p), "survival_probability": round(float(s), 3)}
            for p, s in zip(preds, probas)]