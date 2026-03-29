import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================================
# Page Configuration
# =========================================
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# =========================================
# Load Data and Model
# =========================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_weather.csv")

@st.cache_resource
def load_model():
    model = joblib.load("weather_model.pkl")
    feature_columns = joblib.load("model_features.pkl")
    return model, feature_columns

df = load_data()
model, feature_columns = load_model()

# =========================================
# App Title
# =========================================
st.title("🌦️ Weather Forecasting & Analytics Dashboard")
st.markdown("A machine learning-based system for weather analysis and rain prediction.")

# =========================================
# Sidebar
# =========================================
st.sidebar.header("🔎 Dashboard Filters")

show_full_data = st.sidebar.checkbox("Show Full Dataset", value=False)

if "RainTomorrow" in df.columns:
    rain_filter = st.sidebar.selectbox(
        "Filter by RainTomorrow",
        options=["All", 0, 1]
    )
    if rain_filter != "All":
        df = df[df["RainTomorrow"] == rain_filter]

# =========================================
# KPI Metrics
# =========================================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Max Temp",
    f"{df['MaxTemp'].mean():.2f} °C" if "MaxTemp" in df.columns else "N/A"
)

col2.metric(
    "Average Humidity",
    f"{df['Humidity3pm'].mean():.2f} %" if "Humidity3pm" in df.columns else "N/A"
)

col3.metric(
    "Average Pressure",
    f"{df['Pressure3pm'].mean():.2f}" if "Pressure3pm" in df.columns else "N/A"
)

col4.metric(
    "Average Wind Speed",
    f"{df['WindSpeed3pm'].mean():.2f}" if "WindSpeed3pm" in df.columns else "N/A"
)

# =========================================
# Alert System
# =========================================
st.subheader("🚨 Weather Alerts")

alert_messages = []

if "MaxTemp" in df.columns and df["MaxTemp"].mean() > 35:
    alert_messages.append("🔥 High Temperature Alert")

if "Humidity3pm" in df.columns and df["Humidity3pm"].mean() > 80:
    alert_messages.append("💧 High Humidity Alert")

if "WindSpeed3pm" in df.columns and df["WindSpeed3pm"].mean() > 40:
    alert_messages.append("🌬️ Strong Wind Warning")

if "RainTomorrow" in df.columns and df["RainTomorrow"].mean() > 0.5:
    alert_messages.append("🌧️ High Chance of Rain in the Dataset")

if alert_messages:
    for msg in alert_messages:
        st.warning(msg)
else:
    st.success("✅ Weather conditions look normal based on dataset averages.")

# =========================================
# Charts Section
# =========================================
st.subheader("📈 Weather Analysis")

chart_col1, chart_col2 = st.columns(2)

if "MaxTemp" in df.columns:
    fig1 = px.histogram(
        df,
        x="MaxTemp",
        nbins=30,
        title="Max Temperature Distribution"
    )
    chart_col1.plotly_chart(fig1, use_container_width=True, key="histogram_max_temp")

if "Humidity3pm" in df.columns and "MaxTemp" in df.columns:
    fig2 = px.scatter(
        df,
        x="Humidity3pm",
        y="MaxTemp",
        title="Humidity vs Max Temperature"
    )
    chart_col2.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

# =========================================
# Rain Distribution
# =========================================
if "RainTomorrow" in df.columns:
    st.subheader("🌧️ Rain Tomorrow Distribution")

    rain_counts = df["RainTomorrow"].value_counts().reset_index()
    rain_counts.columns = ["RainTomorrow", "Count"]

    fig3 = px.bar(
        rain_counts,
        x="RainTomorrow",
        y="Count",
        title="RainTomorrow Class Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True, key="rain_distribution")

# =========================================
# Correlation Heatmap
# =========================================
st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig4 = px.imshow(
    corr,
    aspect="auto",
    title="Feature Correlation Heatmap"
)
st.plotly_chart(fig4, use_container_width=True, key="correlation_heatmap")

# =========================================
# Row-Based Prediction
# =========================================
st.subheader("🤖 Row-Based Rain Prediction")

if "RainTomorrow" in df.columns:
    available_features = [col for col in feature_columns if col in df.columns]

    selected_row = st.selectbox(
        "Select a row index for prediction",
        options=df.index.tolist()
    )

    input_row = df.loc[[selected_row], available_features]
    row_prediction = model.predict(input_row)[0]

    if row_prediction == 1:
        st.error("Prediction Result: Rain Expected Tomorrow ☔")
    else:
        st.success("Prediction Result: No Rain Expected Tomorrow ☀️")

# =========================================
# Custom Prediction Section
# =========================================
st.subheader("🧠 Custom Prediction")

col_a, col_b, col_c = st.columns(3)

humidity = col_a.number_input(
    "Humidity3pm",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

pressure = col_b.number_input(
    "Pressure3pm",
    min_value=900.0,
    max_value=1100.0,
    value=1008.0
)

windspeed = col_c.number_input(
    "WindSpeed3pm",
    min_value=0.0,
    max_value=150.0,
    value=25.0
)

if st.button("Predict Rain"):

    # Create full input dataframe with all expected features
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill known values only if columns exist
    if "Humidity3pm" in input_df.columns:
        input_df.at[0, "Humidity3pm"] = humidity

    if "Pressure3pm" in input_df.columns:
        input_df.at[0, "Pressure3pm"] = pressure

    if "WindSpeed3pm" in input_df.columns:
        input_df.at[0, "WindSpeed3pm"] = windspeed

    # Model prediction
    custom_prediction = model.predict(input_df)[0]

    st.write("### 🤖 Model Prediction")
    if custom_prediction == 1:
        st.error("Model Result: Rain Expected Tomorrow ☔")
    else:
        st.success("Model Result: No Rain Expected Tomorrow ☀️")

    # Dataset-based condition
    st.write("### 📊 Dataset-Based Weather Logic")
    if humidity >= 80 and pressure <= 1008 and windspeed >= 25:
        st.warning("Dataset Pattern Result: Higher chance of rain 🌧️")
    elif humidity >= 70 and pressure <= 1012 and windspeed >= 15:
        st.info("Dataset Pattern Result: Moderate chance of rain ⛅")
    else:
        st.success("Dataset Pattern Result: Lower chance of rain ☀️")

    # Input summary
    st.write("### 📝 Input Summary")
    st.markdown(f"""
- **Humidity3pm:** {humidity}
- **Pressure3pm:** {pressure}
- **WindSpeed3pm:** {windspeed}
""")

# =========================================
# Dataset Preview
# =========================================
st.subheader("🗂️ Dataset Preview")

if show_full_data:
    st.dataframe(df)
else:
    st.dataframe(df.head(20))

# =========================================
# Download Button
# =========================================
csv_data = df.to_csv(index=False)

st.download_button(
    label="📥 Download Processed Dataset",
    data=csv_data,
    file_name="processed_weather.csv",
    mime="text/csv"
)

# =========================================
# Insights Section
# =========================================
st.subheader("💡 Key Insights")

insights = []

if "MaxTemp" in df.columns:
    insights.append(f"Average maximum temperature is **{df['MaxTemp'].mean():.2f} °C**.")

if "Humidity3pm" in df.columns:
    insights.append(f"Average humidity at 3 PM is **{df['Humidity3pm'].mean():.2f}%**.")

if "Pressure3pm" in df.columns:
    insights.append(f"Average pressure at 3 PM is **{df['Pressure3pm'].mean():.2f}**.")

if "RainTomorrow" in df.columns:
    rain_percent = df["RainTomorrow"].mean() * 100
    insights.append(f"Rain is present in approximately **{rain_percent:.2f}%** of records.")

for item in insights:
    st.markdown(f"- {item}")

# =========================================
# About Section
# =========================================
st.subheader("📌 About This Project")

st.markdown("""
This project is a **Weather Forecasting & Analytics System** built using historical weather data and machine learning.

### Key Features:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Weather trend visualization
- Rain prediction using a trained ML model
- Interactive Streamlit dashboard
- Custom prediction based on user inputs
""")
