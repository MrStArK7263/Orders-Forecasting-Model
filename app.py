import datetime as dt
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
INPUT_FIELDS = ["non_urgent_orders", "urgent_orders", "fiscal_sector_orders",
                "traffic_controller_orders", "banking_orders_1", "banking_orders_2",
                "banking_orders_3"]

st.set_page_config(page_title="Daily Demand Forecasting", page_icon="📦", layout="wide")
st.title("📦 Daily Demand Forecasting")


def get_week_of_month(d):
    offset = d.replace(day=1).weekday()       # Monday = 0
    return (d.day + offset - 1) // 7 + 1      # same rule as the API / notebook


def next_working_day(d):
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


default_date = next_working_day(dt.date.today() + dt.timedelta(days=1))
tab1, tab2 = st.tabs(["📅 Single day prediction", "📈 7-day forecast"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.write("Pick a date – the week of the month and weekday are filled in for you.")
    date = st.date_input("Date", value=default_date, key="single_date")

    if date.weekday() >= 5:
        st.warning("Weekend selected. The data only contains Monday–Friday, so please choose a working day.")
    else:
        week, day = get_week_of_month(date), date.strftime("%A")
        st.caption(f"Week of month: **{week}**   |   Day: **{day}**")

        col1, col2 = st.columns(2)
        with col1:
            non_urgent = st.number_input("Non-urgent orders", min_value=0.0, value=150.0)
            urgent = st.number_input("Urgent orders", min_value=0.0, value=110.0)
            fiscal = st.number_input("Fiscal sector orders", min_value=0.0, value=10.0)
        with col2:
            traffic = st.number_input("Traffic controller sector orders", min_value=0.0, value=40000.0)
            bank1 = st.number_input("Banking orders (1)", min_value=0.0, value=50000.0)
            bank2 = st.number_input("Banking orders (2)", min_value=0.0, value=70000.0)
            bank3 = st.number_input("Banking orders (3)", min_value=0.0, value=20000.0)

        if st.button("Predict", key="predict_btn"):
            payload = {
                "week_of_month": week, "day_of_week": day,
                "non_urgent_orders": non_urgent, "urgent_orders": urgent,
                "fiscal_sector_orders": fiscal, "traffic_controller_orders": traffic,
                "banking_orders_1": bank1, "banking_orders_2": bank2, "banking_orders_3": bank3,
            }
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                if r.status_code == 200:
                    res = r.json()
                    st.success(f"Predicted total orders: **{res['predicted_total_orders']}**")
                    low, high = res["approx_range"]
                    st.info(f"Typical range: {low} – {high}   |   Model version: {res['model_version']}")
                else:
                    st.error(f"API error: {r.json().get('detail', r.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API at {API_URL}. ({e})")

# ------------------------------------------------------------------ TAB 2
def call_forecast(start, overrides=None):
    body = {"start_date": start.isoformat(), "days": 7}
    if overrides:
        body["overrides"] = overrides
    try:
        r = requests.post(f"{API_URL}/forecast", json=body, timeout=60)
        if r.status_code == 200:
            st.session_state["forecast"] = r.json()
        else:
            st.session_state.pop("forecast", None)
            st.error(f"API error: {r.json().get('detail', r.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. ({e})")


with tab2:
    st.write("Choose a start date on the calendar to forecast the **next 7 working days** "
             "(weekends are skipped because the data has no weekend orders).")
    start = st.date_input("Start date", value=default_date, key="start_date")

    if st.button("Forecast next 7 working days"):
        call_forecast(start)

    fc = st.session_state.get("forecast")
    if fc:
        df = pd.DataFrame(fc["forecast"])
        df["label"] = pd.to_datetime(df["date"]).dt.strftime("%a %d %b")
        df["low"] = df["approx_range"].apply(lambda r: r[0])
        df["high"] = df["approx_range"].apply(lambda r: r[1])

        st.subheader("Forecast")
        st.bar_chart(df.set_index("label")["predicted_total_orders"])
        st.dataframe(
            df[["date", "weekday", "week_of_month", "predicted_total_orders", "low", "high", "inputs_source"]]
            .rename(columns={"predicted_total_orders": "predicted orders", "low": "range low",
                             "high": "range high", "inputs_source": "inputs"}),
            hide_index=True)
        st.caption(f"Model version {fc['model_version']}. Unless you edit them, the other inputs "
                   "are the typical (median) values for that weekday from the historical data, "
                   "so this forecast mainly shows the weekday pattern.")

        with st.expander("✏️ Edit the expected inputs for each day"):
            inputs_df = pd.DataFrame([{"date": r["date"], **r["inputs"]} for r in fc["forecast"]])
            edited = st.data_editor(inputs_df, hide_index=True, disabled=["date"], key="editor")
            if st.button("Recalculate with my inputs"):
                overrides = edited.to_dict(orient="records")
                call_forecast(start, overrides)
                st.rerun()
