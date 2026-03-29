import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Weather Forecasting Dashboard", page_icon="🌦️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("processed_weather_eda.csv")

@st.cache_resource
def load_model():
    return joblib.load("weather_model.pkl")

df = load_data()
model = load_model()

st.title("🌦️ Weather Forecasting & Analytics Dashboard")
st.write("Historical weather data analysis and rain prediction system")

st.subheader("Dataset Preview")
st.dataframe(df.head(20))

if "MaxTemp" in df.columns:
    st.subheader("Max Temperature Distribution")
    fig1 = px.histogram(df, x="MaxTemp", nbins=30, title="Max Temperature Distribution")
    st.plotly_chart(fig1, use_container_width=True, key="hist_max_temp")

if "Humidity3pm" in df.columns and "MaxTemp" in df.columns:
    st.subheader("Humidity vs Max Temperature")
    fig2 = px.scatter(df, x="Humidity3pm", y="MaxTemp", title="Humidity vs Max Temperature")
    st.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

st.subheader("Custom Prediction")

humidity = st.number_input("Humidity3pm", min_value=0.0, max_value=100.0, value=80.0)
pressure = st.number_input("Pressure3pm", min_value=900.0, max_value=1100.0, value=1008.0)
windspeed = st.number_input("WindSpeed3pm", min_value=0.0, max_value=150.0, value=25.0)

if st.button("Predict Rain"):
    feature_columns = joblib.load("model_features.pkl")
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    if "Humidity3pm" in input_df.columns:
        input_df.at[0, "Humidity3pm"] = humidity
    if "Pressure3pm" in input_df.columns:
        input_df.at[0, "Pressure3pm"] = pressure
    if "WindSpeed3pm" in input_df.columns:
        input_df.at[0, "WindSpeed3pm"] = windspeed

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Rain Expected Tomorrow ☔")
    else:
        st.success("No Rain Expected Tomorrow ☀️")
