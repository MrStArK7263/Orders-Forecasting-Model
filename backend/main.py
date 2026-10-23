import datetime as dt
import os
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Settings (from environment variables) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ---------- Load the saved model once, when the server starts ----------
try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    bundle = None
    print(f"Could not load model: {e}")

app = FastAPI(title="Daily Demand Forecasting API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


# ---------- Input validation ----------
class OrderInput(BaseModel):
    week_of_month: int = Field(..., ge=1, le=5, description="1 to 5")
    day_of_week: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    non_urgent_orders: float = Field(..., ge=0)
    urgent_orders: float = Field(..., ge=0)
    fiscal_sector_orders: float = Field(..., ge=0)
    traffic_controller_orders: float = Field(..., ge=0)
    banking_orders_1: float = Field(..., ge=0)
    banking_orders_2: float = Field(..., ge=0)
    banking_orders_3: float = Field(..., ge=0)


class ForecastDayInput(BaseModel):
    """Optional user-edited inputs for one future date."""
    date: dt.date
    non_urgent_orders: float = Field(..., ge=0)
    urgent_orders: float = Field(..., ge=0)
    fiscal_sector_orders: float = Field(..., ge=0)
    traffic_controller_orders: float = Field(..., ge=0)
    banking_orders_1: float = Field(..., ge=0)
    banking_orders_2: float = Field(..., ge=0)
    banking_orders_3: float = Field(..., ge=0)


class ForecastRequest(BaseModel):
    start_date: dt.date
    days: int = Field(7, ge=1, le=14, description="number of WORKING days to forecast")
    overrides: Optional[List[ForecastDayInput]] = None


INPUT_FIELDS = ["non_urgent_orders", "urgent_orders", "fiscal_sector_orders",
                "traffic_controller_orders", "banking_orders_1", "banking_orders_2",
                "banking_orders_3"]


# ---------- Calendar helpers ----------
def get_week_of_month(d: dt.date) -> int:
    offset = d.replace(day=1).weekday()          # Monday = 0
    return (d.day + offset - 1) // 7 + 1


def next_working_days(start: dt.date, n: int):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:                      # skip Saturday / Sunday
            days.append(d)
        d += dt.timedelta(days=1)
    return days


# ---------- Same preprocessing as the notebook ----------
def prepare_input(data: dict) -> pd.DataFrame:
    row = pd.DataFrame([data])
    row["urgent_ratio"] = row["urgent_orders"] / (row["urgent_orders"] + row["non_urgent_orders"])
    row["banking_total"] = row["banking_orders_1"] + row["banking_orders_2"] + row["banking_orders_3"]
    row["fiscal_log"] = np.log1p(row["fiscal_sector_orders"])

    full = pd.DataFrame(0, index=row.index, columns=bundle["all_columns"])
    for col in row.columns:
        if col in full.columns:
            full[col] = row[col]

    for c in (f"week_of_month_{data['week_of_month']}", f"day_name_{data['day_of_week']}"):
        if c in full.columns:       # week 1 / Monday are the baseline -> no column
            full[c] = 1

    full[bundle["scale_cols"]] = bundle["scaler"].transform(full[bundle["scale_cols"]])
    return full[bundle["selected_features"]]


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok" if bundle is not None else "model_not_loaded"}


@app.get("/model-info")
def model_info():
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model_version": bundle["model_version"],
            "model_type": type(bundle["model"]).__name__,
            "test_mae": round(bundle["test_mae"], 2),
            "features_used": bundle["selected_features"]}


@app.post("/predict")
def predict(item: OrderInput):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if item.urgent_orders + item.non_urgent_orders == 0:
        raise HTTPException(status_code=422,
                            detail="urgent_orders + non_urgent_orders must be greater than 0")
    try:
        x = prepare_input(item.model_dump())
        pred = float(bundle["model"].predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"predicted_total_orders": round(max(pred, 0), 2),
            "approx_range": [round(max(pred - bundle["test_mae"], 0), 2),
                             round(pred + bundle["test_mae"], 2)],
            "model_version": bundle["model_version"]}


@app.get("/calendar-info")
def calendar_info(date: dt.date):
    """Week of month and weekday for a calendar date."""
    return {"date": date.isoformat(), "weekday": date.strftime("%A"),
            "week_of_month": get_week_of_month(date), "is_weekend": date.weekday() >= 5}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if "weekday_profile" not in bundle:
        raise HTTPException(status_code=503,
                            detail="This model.pkl has no weekday_profile. Re-run the notebook and redeploy.")
    user_inputs = {o.date: o for o in (req.overrides or [])}
    results = []
    try:
        for d in next_working_days(req.start_date, req.days):
            weekday = d.strftime("%A")
            if d in user_inputs:
                inputs = {k: getattr(user_inputs[d], k) for k in INPUT_FIELDS}
                source = "user"
            else:
                inputs = {k: float(bundle["weekday_profile"][weekday][k]) for k in INPUT_FIELDS}
                source = "typical"
            if inputs["urgent_orders"] + inputs["non_urgent_orders"] == 0:
                raise HTTPException(status_code=422,
                                    detail=f"{d}: urgent + non-urgent orders must be greater than 0")
            week = get_week_of_month(d)
            x = prepare_input({"week_of_month": week, "day_of_week": weekday, **inputs})
            pred = max(float(bundle["model"].predict(x)[0]), 0)
            results.append({"date": d.isoformat(), "weekday": weekday, "week_of_month": week,
                            "predicted_total_orders": round(pred, 1),
                            "approx_range": [round(max(pred - bundle["test_mae"], 0), 1),
                                             round(pred + bundle["test_mae"], 1)],
                            "inputs_source": source, "inputs": inputs})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")
    return {"model_version": bundle["model_version"], "forecast": results}
