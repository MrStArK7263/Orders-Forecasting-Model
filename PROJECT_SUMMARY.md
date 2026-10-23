# Project Summary – Daily Demand Forecasting

## Project flow
Raw CSV -> understand -> clean -> EDA -> correlation -> feature engineering -> encoding -> split + scale -> feature selection -> train 4 models -> evaluate -> tune -> choose -> save (joblib) -> FastAPI -> Streamlit -> Docker -> Render

Runtime flow: User -> Streamlit form -> POST /predict (FastAPI) -> validation (Pydantic) -> same preprocessing as notebook -> Ridge model -> JSON -> shown on UI

## What is happening behind each step
1. **Understanding** – 60 rows, 13 columns + a useless index. Target `Total orders` is a number, so this is **regression**, not classification (the assignment's churn / confusion-matrix wording is generic).
2. **Cleaning** – no missing, duplicate or invalid values. Outliers (esp. fiscal sector, banking 1) are kept: only 60 rows and the spikes are real. Row index dropped. **Leakage:** order_type_a + b + c == target exactly, so they are removed.
3. **EDA** – demand is right-skewed; Monday is the busiest day (~390) and Friday the quietest (~263); non-urgent and banking-2 orders move with demand.
4. **Correlation** – non_urgent 0.93, banking_2 0.80, urgent 0.73 with target; fiscal and banking_3 ~0. Highest feature pair is 0.79 so nothing is removed for multicollinearity.
5. **Feature engineering** – urgent_ratio, banking_total, log of fiscal (skew 3.15 -> 0.63).
6. **Encoding (separate)** – week and day are categories -> one-hot, drop_first=True (baseline = week 1, Monday) to avoid the dummy-variable trap.
7. **Split + scale** – 80/20 split; StandardScaler fitted on train only (prevents leakage).
8. **Feature selection** – mutual information + random-forest importance; 17 -> 6 features (the raw banking columns are removed because banking_total already contains them), CV score did not drop (LR 0.953 -> 0.966).
9. **Models** – Linear, Ridge, Decision Tree, Random Forest.
10. **Evaluation** – MAE, RMSE, R², actual-vs-predicted and residual plots (these replace confusion matrix / precision / recall / ROC, which only exist for classification).
11. **Tuning** – GridSearchCV on Ridge (alpha) and Random Forest; improvement is small, so tuning confirmed rather than changed the ranking.
12. **Final choice** – tuned Ridge: CV R² ≈ 0.97, MAE ≈ 21 orders, ~1 ms inference, simple/explainable. Not picked on one test number: the test set is only 12 rows.
13. **Calendar + 7-day forecast** – the data has no dates, so for a calendar date we compute weekday and week-of-month (Monday-based weeks; week 1 may be partial, matching the data). Future inputs are unknown, so each future day uses the historical *median* of every input for that weekday (stored in model.pkl as `weekday_profile`) and the user may edit them. Only Monday–Friday are forecast (no weekend data). It is a scenario forecast that mostly reflects the weekday pattern: Monday ≈ 373 orders, Friday ≈ 249.
14. **Save/API/UI/Docker** – bundle = model + scaler + column order + version; API repeats the notebook preprocessing exactly.

## New API endpoints
- `GET /calendar-info?date=YYYY-MM-DD` – weekday + week of month
- `POST /forecast` – body `{"start_date": "2026-10-26", "days": 7, "overrides": [ ...optional edited inputs per date... ]}`

## Encoding table (drop_first=True; Week 1 and Monday are the all-zero baseline)

| week_of_month | week_of_month_2 | week_of_month_3 | week_of_month_4 | week_of_month_5 |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 1 | 0 | 0 | 0 |
| 3 | 0 | 1 | 0 | 0 |
| 4 | 0 | 0 | 1 | 0 |
| 5 | 0 | 0 | 0 | 1 |

| day (code) | day_name_Tuesday | day_name_Wednesday | day_name_Thursday | day_name_Friday |
|---|---|---|---|---|
| Monday (2) | 0 | 0 | 0 | 0 |
| Tuesday (3) | 1 | 0 | 0 | 0 |
| Wednesday (4) | 0 | 1 | 0 | 0 |
| Thursday (5) | 0 | 0 | 1 | 0 |
| Friday (6) | 0 | 0 | 0 | 1 |
