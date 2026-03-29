import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------------
# Load data and model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"processed_weather_eda.csv")

@st.cache_resource
def load_model():
    return joblib.load(r"weather_model.pkl")

feature_columns = joblib.load(r"model_features.pkl")

df = load_data()
model = load_model()

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

if "RainTomorrow" in df.columns:
    rain_filter = st.sidebar.selectbox(
        "Filter by RainTomorrow",
        options=["All", 0, 1]
    )
    if rain_filter != "All":
        df = df[df["RainTomorrow"] == rain_filter]

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

if "MaxTemp" in df.columns:
    col1.metric("Average Max Temp", f"{df['MaxTemp'].mean():.2f} °C")
else:
    col1.metric("Average Max Temp", "N/A")

if "Humidity3pm" in df.columns:
    col2.metric("Average Humidity", f"{df['Humidity3pm'].mean():.2f} %")
else:
    col2.metric("Average Humidity", "N/A")

if "Pressure3pm" in df.columns:
    col3.metric("Average Pressure", f"{df['Pressure3pm'].mean():.2f}")
else:
    col3.metric("Average Pressure", "N/A")

if "WindSpeed3pm" in df.columns:
    col4.metric("Average Wind Speed", f"{df['WindSpeed3pm'].mean():.2f}")
else:
    col4.metric("Average Wind Speed", "N/A")


#weather alerts based on thresholds


st.subheader("🚨 Weather Alerts")

if df['MaxTemp'].mean() > 35:
    st.warning("High Temperature Alert 🔥")

if df['Humidity3pm'].mean() > 80:
    st.warning("High Humidity Alert 💧")

# -------------------------------
# Charts
# -------------------------------
st.subheader("📈 Weather Analysis")

chart_col1, chart_col2 = st.columns(2)

if "MaxTemp" in df.columns:
    fig1 = px.histogram(
        df,
        x="MaxTemp",
        nbins=30,
        title="Max Temperature Distribution"
    )
    chart_col1.plotly_chart(fig1, use_container_width=True, key="hist_max_temp")

if "Humidity3pm" in df.columns and "MaxTemp" in df.columns:
    fig2 = px.scatter(
        df,
        x="Humidity3pm",
        y="MaxTemp",
        title="Humidity vs Max Temperature"
    )
    chart_col2.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

# -------------------------------
# RainTomorrow Distribution
# -------------------------------
if "RainTomorrow" in df.columns:
    st.subheader("🌧️ Rain Tomorrow Distribution")

    rain_counts = df["RainTomorrow"].value_counts().reset_index()
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

corr = df.corr(numeric_only=True)

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

if "RainTomorrow" in df.columns:
    feature_columns = [col for col in df.columns if col != "RainTomorrow"]

    selected_row = st.selectbox(
        "Select a row index for prediction",
        options=df.index.tolist()
    )

    input_data = df.loc[[selected_row], feature_columns]
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

# Load feature columns
feature_columns = joblib.load("model_features.pkl")

col1, col2, col3 = st.columns(3)

humidity = col1.number_input("Humidity3pm", min_value=0.0, max_value=100.0, value=80.0)
pressure = col2.number_input("Pressure3pm", min_value=900.0, max_value=1100.0, value=1008.0)
windspeed = col3.number_input("WindSpeed3pm", min_value=0.0, max_value=150.0, value=25.0)

if st.button("Predict Rain"):

    # Create full input dataframe with all model features
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # Fill user input values
    if "Humidity3pm" in input_df.columns:
        input_df.at[0, "Humidity3pm"] = humidity

    if "Pressure3pm" in input_df.columns:
        input_df.at[0, "Pressure3pm"] = pressure

    if "WindSpeed3pm" in input_df.columns:
        input_df.at[0, "WindSpeed3pm"] = windspeed

    # -------------------------------
    # Model Prediction
    # -------------------------------
    prediction = model.predict(input_df)[0]

    st.write("### 🤖 Model Prediction")
    if prediction == 1:
        st.error("Model Result: Rain Expected Tomorrow ☔")
    else:
        st.success("Model Result: No Rain Expected Tomorrow ☀️")

    # -------------------------------
    # Dataset-Based Rule Prediction
    # -------------------------------
    st.write("### 📊 Dataset-Based Weather Logic")

    if humidity >= 80 and pressure <= 1008 and windspeed >= 25:
        st.warning("Dataset Pattern Result: Higher chance of rain 🌧️")
    elif humidity >= 70 and pressure <= 1012 and windspeed >= 15:
        st.info("Dataset Pattern Result: Moderate chance of rain ⛅")
    else:
        st.success("Dataset Pattern Result: Lower chance of rain ☀️")

    # -------------------------------
    # Input Summary
    # -------------------------------
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
    st.dataframe(df)
else:
    st.dataframe(df.head(20))

# -------------------------------
# Insights Section
# -------------------------------
st.subheader("💡 Key Insights")

insights = []

if "MaxTemp" in df.columns:
    insights.append(f"Average maximum temperature is **{df['MaxTemp'].mean():.2f} °C**.")

if "Humidity3pm" in df.columns:
    insights.append(f"Average humidity at 3 PM is **{df['Humidity3pm'].mean():.2f}%**.")

if "RainTomorrow" in df.columns:
    rain_percentage = df["RainTomorrow"].mean() * 100
    insights.append(f"Rain is expected in approximately **{rain_percentage:.2f}%** of records.")

if "Pressure3pm" in df.columns:
    insights.append(f"Average pressure at 3 PM is **{df['Pressure3pm'].mean():.2f}**.")

if insights:
    for insight in insights:
        st.markdown(f"- {insight}")
else:
    st.write("No insights available from current dataset.")
