import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "data_processed" / "final_covid_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "epidemic_prediction_model.pkl"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Check your data folder.")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.copy()
    if "mobility" in df.columns:
        df["mobility"] = pd.to_numeric(df["mobility"], errors="coerce")
    for c in ["cases", "deaths", "vaccination_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["country"] = df["country"].astype(str)
    return df


@st.cache_data
def get_latest_country_frame(df, country):
    cdf = df[df["country"] == country].sort_values("date").copy()
    if cdf.empty:
        return cdf
    cdf["daily_cases"] = cdf["cases"].diff().fillna(cdf["cases"])  # assuming cumulative if so
    cdf["daily_cases"] = cdf["daily_cases"].clip(lower=0)
    cdf["daily_vaccination"] = cdf["vaccination_rate"].diff().fillna(0)
    return cdf


def calculate_risk_level(value):
    if pd.isna(value):
        return "Low"
    if value >= 10000:
        return "High"
    elif value >= 2000:
        return "Medium"
    else:
        return "Low"


def run_prediction(df, country):
    cdf = get_latest_country_frame(df, country)
    if cdf.empty:
        return pd.DataFrame()

    last_14 = cdf.tail(14)
    # baseline simple prediction if model absent or fails
    predicted = None

    if MODEL_PATH.exists():
        try:
            import pickle
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)

            # attempt for generic Scikit and series
            X_future = np.arange(len(cdf), len(cdf) + 14).reshape(-1, 1)
            predicted = model.predict(X_future)
            predicted = np.maximum(predicted, 0)
        except Exception as e:
            st.warning(f"Model loaded but prediction step failed: {e}. Using fallback linear trend.")

    if predicted is None:
        y = last_14["daily_cases"].astype(float).values
        if len(y) < 2:
            predicted = np.repeat(last_14["cases"].iloc[-1], 14)
        else:
            x = np.arange(len(y))
            coef = np.polyfit(x, y, deg=1)
            trend = coef[0] * np.arange(len(y), len(y) + 14) + coef[1]
            predicted = np.maximum(trend, 0)

    future_dates = [cdf["date"].max() + timedelta(days=i + 1) for i in range(14)]
    return pd.DataFrame({"date": future_dates, "predicted_cases": predicted})


def page_global_overview(df):
    st.header("Global Overview")

    total_cases = int(df["cases"].max()) if "cases" in df.columns else 0
    total_deaths = int(df["deaths"].max()) if "deaths" in df.columns else 0
    global_vax = df.groupby("date")["vaccination_rate"].sum().max() if "vaccination_rate" in df.columns else 0

    st.metric("Total Global Cases", f"{total_cases:,}")
    st.metric("Total Global Deaths", f"{total_deaths:,}")
    st.metric("Peak Global Vaccination", f"{global_vax:,.0f}")

    latest = df[df["date"] == df["date"].max()] if not df.empty else pd.DataFrame()
    if not latest.empty:
        country_recent = latest.groupby("country")["cases"].max().reset_index()
        def choropleth_dataframe():
            cd = country_recent.copy()
            cd.columns = ["location", "cases"]
            return cd

        map_df = choropleth_dataframe()
        fig_map = px.choropleth(
            map_df,
            locations="location",
            locationmode="country names",
            color="cases",
            hover_name="location",
            color_continuous_scale="Reds",
            title="Global Outbreak Map (latest date)",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    trend = df.groupby("date")["cases"].sum().reset_index()
    if not trend.empty:
        fig_trend = px.line(trend, x="date", y="cases", title="Global Total Cases Trend")
        st.plotly_chart(fig_trend, use_container_width=True)


def page_country_analysis(df):
    st.header("Country Analysis")

    countries = sorted(df["country"].unique())
    country = st.selectbox("Select Country", countries, index=countries.index("India") if "India" in countries else 0)

    cdf = get_latest_country_frame(df, country)
    if cdf.empty:
        st.warning("No data for selected country")
        return

    st.subheader(f"Historical Cases: {country}")
    fig_cases = px.line(cdf, x="date", y="cases", title=f"Cumulative Cases ({country})")
    st.plotly_chart(fig_cases, use_container_width=True)

    st.subheader("Vaccination Statistics")
    if "vaccination_rate" in cdf.columns:
        fig_vax = px.line(cdf, x="date", y="vaccination_rate", title=f"Vaccination Rate ({country})")
        st.plotly_chart(fig_vax, use_container_width=True)
    else:
        st.info("No vaccination data available for this country.")

    st.subheader("Mobility Changes")
    if "mobility" in cdf.columns:
        fig_mob = px.line(cdf, x="date", y="mobility", title=f"Mobility Changes ({country})")
        st.plotly_chart(fig_mob, use_container_width=True)
    else:
        st.info("No mobility data available for this country.")


def page_prediction_dashboard(df):
    st.header("Prediction Dashboard")

    countries = sorted(df["country"].unique())
    country = st.selectbox("Select Country for Prediction", countries, index=countries.index("India") if "India" in countries else 0)
    prediction_df = run_prediction(df, country)
    if prediction_df.empty:
        st.warning("Unable to produce prediction for selected country")
        return

    st.subheader("Predicted Cases Next 14 Days")
    st.dataframe(prediction_df.set_index("date"))

    cdf = get_latest_country_frame(df, country)
    if not cdf.empty:
        last_actual = cdf.tail(30)[["date", "daily_cases"]].rename(columns={"daily_cases": "cases"})
        fig = px.line(last_actual, x="date", y="cases", labels={"cases": "Daily Cases"}, title="Actual vs Predicted Daily Cases")
        fig.add_scatter(x=prediction_df["date"], y=prediction_df["predicted_cases"], mode="lines+markers", name="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    if MODEL_PATH.exists():
        st.success("Loaded prediction model from epidemic_prediction_model.pkl")
    else:
        st.warning("epidemic_prediction_model.pkl not found. Fallback model used.")


def page_risk_map(df):
    st.header("Risk Map")

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].groupby("country")["cases"].sum().reset_index()
    latest = latest.rename(columns={"cases": "total_cases"})

    # using last 7-day average of daily cases by country.
    recent = df[df["date"] >= (latest_date - timedelta(days=7))].copy()
    recent = recent.groupby("country")["cases"].sum().reset_index()
    recent["daily_avg"] = recent["cases"] / 7.0
    recent["risk_level"] = recent["daily_avg"].apply(calculate_risk_level)

    risk_colors = {"High": "red", "Medium": "orange", "Low": "green"}

    fig = px.choropleth(
        recent,
        locations="country",
        locationmode="country names",
        color="risk_level",
        color_discrete_map=risk_colors,
        hover_name="country",
        hover_data={"daily_avg": ":.1f", "risk_level": True},
        title="World Risk Levels (7-day avg daily cases)",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Covid Dashboard", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select page", ["Global Overview", "Country Analysis", "Prediction Dashboard", "Risk Map"])

    df = load_data()
    if df.empty:
        st.stop()

    if page == "Global Overview":
        page_global_overview(df)
    elif page == "Country Analysis":
        page_country_analysis(df)
    elif page == "Prediction Dashboard":
        page_prediction_dashboard(df)
    elif page == "Risk Map":
        page_risk_map(df)


if __name__ == "__main__":
    main()
