import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------------
# File Paths (Deploy-Safe)
# -------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_weather_eda.csv"
MODEL_PATH = BASE_DIR / "weather_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

# -------------------------------
# Load Data and Model
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

# -------------------------------
# Safe App Loading
# -------------------------------
try:
    df = load_data()
    model = load_model()
    feature_columns_saved = load_feature_columns()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure these files are in the same folder as app.py:")
    st.code(
        "processed_weather_eda.csv\nweather_model.pkl\nmodel_features.pkl",
        language="text"
    )
    st.stop()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

show_data = st.sidebar.checkbox("Show full dataset preview", value=False)

filtered_df = df.copy()

if "RainTomorrow" in filtered_df.columns:
    rain_filter = st.sidebar.selectbox(
        "🌧️ Filter by RainTomorrow",
        options=["All", 0, 1]
    )
    if rain_filter != "All":
        filtered_df = filtered_df[filtered_df["RainTomorrow"] == rain_filter]

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🌦️ Weather Forecasting Dashboard
</h1>
<p style='text-align: center; font-size:18px;'>
Historical Weather Analysis + Rain Prediction System
</p>
""", unsafe_allow_html=True)

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🌡️ Avg Temp",
    f"{filtered_df['MaxTemp'].mean():.1f}°C" if "MaxTemp" in filtered_df.columns else "N/A"
)
col2.metric(
    "💧 Humidity",
    f"{filtered_df['Humidity3pm'].mean():.1f}%" if "Humidity3pm" in filtered_df.columns else "N/A"
)
col3.metric(
    "🌬️ Wind",
    f"{filtered_df['WindSpeed3pm'].mean():.1f}" if "WindSpeed3pm" in filtered_df.columns else "N/A"
)
col4.metric(
    "🌧️ Rain %",
    f"{filtered_df['RainTomorrow'].mean() * 100:.1f}%" if "RainTomorrow" in filtered_df.columns else "N/A"
)

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

with chart_col1:
    if "MaxTemp" in filtered_df.columns:
        fig1 = px.histogram(
            filtered_df,
            x="MaxTemp",
            nbins=30,
            title="Temperature Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True, key="hist_max_temp")

with chart_col2:
    if "Humidity3pm" in filtered_df.columns and "MaxTemp" in filtered_df.columns:
        fig2 = px.scatter(
            filtered_df,
            x="Humidity3pm",
            y="MaxTemp",
            color="RainTomorrow" if "RainTomorrow" in filtered_df.columns else None,
            title="Humidity vs Temperature"
        )
        st.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

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
# Feature Importance
# -------------------------------
st.subheader("🧠 Feature Importance")

if hasattr(model, "feature_importances_"):
    try:
        feat_df = pd.DataFrame({
            "Feature": feature_columns_saved,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig4 = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Important Features"
        )
        st.plotly_chart(fig4, use_container_width=True, key="feature_importance_chart")
    except Exception as e:
        st.info(f"Feature importance could not be displayed: {e}")
else:
    st.info("Feature importance is not available for this model.")

# -------------------------------
# Prediction from Existing Dataset Row
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
# Custom Prediction
# -------------------------------
st.subheader("🔮 Custom Rain Prediction")

col5, col6, col7 = st.columns(3)

humidity = col5.number_input("💧 Humidity3pm", min_value=0.0, max_value=100.0, value=80.0)
pressure = col6.number_input("🌡️ Pressure3pm", min_value=900.0, max_value=1100.0, value=1008.0)
windspeed = col7.number_input("🌬️ WindSpeed3pm", min_value=0.0, max_value=150.0, value=25.0)

if st.button("🚀 Predict Rain"):
    try:
        input_df = pd.DataFrame(columns=feature_columns_saved)
        input_df.loc[0] = 0

        if "Humidity3pm" in input_df.columns:
            input_df.at[0, "Humidity3pm"] = humidity
        if "Pressure3pm" in input_df.columns:
            input_df.at[0, "Pressure3pm"] = pressure
        if "WindSpeed3pm" in input_df.columns:
            input_df.at[0, "WindSpeed3pm"] = windspeed

        prediction = model.predict(input_df)[0]

        st.write("### 🤖 Model Prediction")
        if prediction == 1:
            st.error("Model Result: Rain Expected Tomorrow ☔")
        else:
            st.success("Model Result: No Rain Expected Tomorrow ☀️")

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
    except Exception as e:
        st.error(f"Prediction failed: {e}")

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
    st.info("\n\n".join(insights))
else:
    st.write("No insights available from current dataset.")
