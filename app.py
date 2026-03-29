import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------------
# File paths (deployment-safe)
# -------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_weather_eda.csv"
MODEL_PATH = BASE_DIR / "weather_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

# -------------------------------
# Load data and model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_feature_columns():
    return joblib.load(FEATURES_PATH)

df = load_data()
model = load_model()
feature_columns_saved = load_feature_columns()

# -------------------------------
# Title
# -------------------------------
st.title("🌦️ Weather Forecasting & Analytics Dashboard")
st.markdown("Historical weather data analysis and rain prediction system")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Dashboard Filters")
show_data = st.sidebar.checkbox("Show full dataset preview", value=False)

filtered_df = df.copy()

if "RainTomorrow" in filtered_df.columns:
    rain_filter = st.sidebar.selectbox(
        "Filter by RainTomorrow",
        options=["All", 0, 1]
    )
    if rain_filter != "All":
        filtered_df = filtered_df[filtered_df["RainTomorrow"] == rain_filter]

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

if "MaxTemp" in filtered_df.columns:
    col1.metric("Average Max Temp", f"{filtered_df['MaxTemp'].mean():.2f} °C")
else:
    col1.metric("Average Max Temp", "N/A")

if "Humidity3pm" in filtered_df.columns:
    col2.metric("Average Humidity", f"{filtered_df['Humidity3pm'].mean():.2f} %")
else:
    col2.metric("Average Humidity", "N/A")

if "Pressure3pm" in filtered_df.columns:
    col3.metric("Average Pressure", f"{filtered_df['Pressure3pm'].mean():.2f}")
else:
    col3.metric("Average Pressure", "N/A")

if "WindSpeed3pm" in filtered_df.columns:
    col4.metric("Average Wind Speed", f"{filtered_df['WindSpeed3pm'].mean():.2f}")
else:
    col4.metric("Average Wind Speed", "N/A")

# -------------------------------
# Weather Alerts
# -------------------------------
st.subheader("🚨 Weather Alerts")

if "MaxTemp" in filtered_df.columns and filtered_df["MaxTemp"].mean() > 35:
    st.warning("High Temperature Alert 🔥")

if "Humidity3pm" in filtered_df.columns and filtered_df["Humidity3pm"].mean() > 80:
    st.warning("High Humidity Alert 💧")

# -------------------------------
# Charts
# -------------------------------
st.subheader("📈 Weather Analysis")
chart_col1, chart_col2 = st.columns(2)

if "MaxTemp" in filtered_df.columns:
    fig1 = px.histogram(
        filtered_df,
        x="MaxTemp",
        nbins=30,
        title="Max Temperature Distribution"
    )
    chart_col1.plotly_chart(fig1, use_container_width=True, key="hist_max_temp")

if "Humidity3pm" in filtered_df.columns and "MaxTemp" in filtered_df.columns:
    fig2 = px.scatter(
        filtered_df,
        x="Humidity3pm",
        y="MaxTemp",
        title="Humidity vs Max Temperature"
    )
    chart_col2.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

# -------------------------------
# RainTomorrow Distribution
# -------------------------------
if "RainTomorrow" in filtered_df.columns:
    st.subheader("🌧️ Rain Tomorrow Distribution")

    rain_counts = filtered_df["RainTomorrow"].value_counts().reset_index()
    rain_counts.columns = ["RainTomorrow", "Count"]

    fig3 = px.bar(
        rain_counts,
        x="RainTomorrow",
        y="Count",
        title="Rain Tomorrow Count"
    )
    st.plotly_chart(fig3, use_container_width=True, key="rain_tomorrow_bar")

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("🔥 Correlation Heatmap")
corr = filtered_df.corr(numeric_only=True)

fig4 = px.imshow(
    corr,
    aspect="auto",
    title="Feature Correlation Heatmap"
)
st.plotly_chart(fig4, use_container_width=True, key="corr_heatmap")

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🤖 Rain Prediction")

if "RainTomorrow" in filtered_df.columns:
    prediction_features = [col for col in filtered_df.columns if col != "RainTomorrow"]

    selected_row = st.selectbox(
        "Select a row index for prediction",
        options=filtered_df.index.tolist()
    )

    input_data = filtered_df.loc[[selected_row], prediction_features]
    prediction = model.predict(input_data)[0]

    st.write("### Prediction Result")
    if prediction == 1:
        st.error("Rain Expected Tomorrow ☔")
    else:
        st.success("No Rain Expected Tomorrow ☀️")
else:
    st.warning("RainTomorrow column not found in dataset.")

# -------------------------------
# Custom Prediction Section
# -------------------------------
st.subheader("🧠 Custom Prediction")

col5, col6, col7 = st.columns(3)

humidity = col5.number_input("Humidity3pm", min_value=0.0, max_value=100.0, value=80.0)
pressure = col6.number_input("Pressure3pm", min_value=900.0, max_value=1100.0, value=1008.0)
windspeed = col7.number_input("WindSpeed3pm", min_value=0.0, max_value=150.0, value=25.0)

if st.button("Predict Rain"):
    input_df = pd.DataFrame(columns=feature_columns_saved)
    input_df.loc[0] = 0

    if "Humidity3pm" in input_df.columns:
        input_df.at[0, "Humidity3pm"] = humidity
    if "Pressure3pm" in input_df.columns:
        input_df.at[0, "Pressure3pm"] = pressure
    if "WindSpeed3pm" in input_df.columns:
        input_df.at[0, "WindSpeed3pm"] = windspeed

    prediction = model.predict(input_df)[0]


    st.write("### 📊 Dataset-Based Weather Logic")
    if humidity >= 80 and pressure <= 1008 and windspeed >= 25:
        st.warning("Dataset Pattern Result: Higher chance of rain 🌧️")
    elif humidity >= 70 and pressure <= 1012 and windspeed >= 15:
        st.info("Dataset Pattern Result: Moderate chance of rain ⛅")
    else:
        st.success("Dataset Pattern Result: Lower chance of rain ☀️")

    st.write("### 📝 Input Summary")
    st.markdown(f"""
- **Humidity3pm:** {humidity}
- **Pressure3pm:** {pressure}
- **WindSpeed3pm:** {windspeed}
""")

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("🗂️ Dataset Preview")

if show_data:
    st.dataframe(filtered_df)
else:
    st.dataframe(filtered_df.head(20))

# -------------------------------
# Insights Section
# -------------------------------
st.subheader("💡 Key Insights")

insights = []

if "MaxTemp" in filtered_df.columns:
    insights.append(f"Average maximum temperature is **{filtered_df['MaxTemp'].mean():.2f} °C**.")

if "Humidity3pm" in filtered_df.columns:
    insights.append(f"Average humidity at 3 PM is **{filtered_df['Humidity3pm'].mean():.2f}%**.")

if "RainTomorrow" in filtered_df.columns:
    rain_percentage = filtered_df["RainTomorrow"].mean() * 100
    insights.append(f"Rain is expected in approximately **{rain_percentage:.2f}%** of records.")

if "Pressure3pm" in filtered_df.columns:
    insights.append(f"Average pressure at 3 PM is **{filtered_df['Pressure3pm'].mean():.2f}**.")

if insights:
    for insight in insights:
        st.markdown(f"- {insight}")
else:
    st.write("No insights available from current dataset.")
