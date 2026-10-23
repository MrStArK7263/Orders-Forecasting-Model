# Daily Demand Forecasting – Deployment Guide

Folder layout
```
Demand_Forecasting_Notebook.ipynb   # all ML steps, one segment per cell
Daily_Demand_Forecasting_Orders.csv
backend/   main.py, model.pkl, requirements.txt, Dockerfile, .env.example
frontend/  app.py, requirements.txt
```

## 1. Train
Open the notebook, run all cells -> it creates `model.pkl`. Copy it into `backend/`.
IMPORTANT: run `pip show scikit-learn pandas numpy joblib` in the same environment as the notebook and put those exact versions in `backend/requirements.txt` (a model saved with one scikit-learn version may not load in another).

## 2. Run the backend locally
```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs  (endpoints: GET /health, GET /model-info, GET /calendar-info, POST /predict, POST /forecast)

## 3. Run the UI locally
```
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
(set `API_URL` if the backend is not on 127.0.0.1:8000)

## 4. Docker
```
cd backend
docker build -t demand-api .
docker run -p 8000:8000 demand-api
```

## 5. Deploy the backend on Render
1. Push the `backend/` folder to a GitHub repo.
2. Render -> New -> Web Service -> select the repo -> Environment: **Docker**.
3. Add environment variables: `MODEL_PATH=model.pkl`, `ALLOWED_ORIGINS=*`.
4. Health check path: `/health`. Deploy and copy the URL (https://xxxx.onrender.com).
(Free Render services sleep when idle, so the first request can take ~50 s.)

## 6. Deploy the Streamlit UI
Push `frontend/` to GitHub and deploy on Streamlit Community Cloud (or a second Render service), and set the environment variable / secret `API_URL=https://xxxx.onrender.com`.

## Model versioning
The version lives inside `model.pkl` (`model_version`) and is returned by `/model-info` and `/predict`. For a new model, change it in the notebook (e.g. 1.1.0), retrain, redeploy.

## Updating an already-deployed app
Re-run the notebook (it creates a new model.pkl with `weekday_profile`), copy model.pkl + main.py into the backend repo, push, and redeploy on Render. Then push the new frontend/app.py. An OLD model.pkl will not work for /forecast (the API says so).
