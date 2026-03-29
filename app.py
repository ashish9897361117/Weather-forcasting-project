import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>
/* Main App */
.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
}

/* Header */
.main-header {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}
.main-header h1 {
    color: #1f4e79;
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #4f6b8a;
    font-size: 1.1rem;
    margin-top: 0;
}

/* KPI cards */
.kpi-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    border-left: 6px solid #4CAF50;
    margin-bottom: 10px;
}
.kpi-title {
    font-size: 0.95rem;
    color: #5c6b7a;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1f2937;
}

/* Section block */
.block-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
    padding: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Deploy-safe file paths
# ---------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_weather_eda.csv"
MODEL_PATH = BASE_DIR / "weather_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

# ---------------------------------
# Load functions
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_feature_columns():
    return joblib.load(FEATURES_PATH)

# ---------------------------------
# Safe loading
# ---------------------------------
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

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

show_full_data = st.sidebar.checkbox("Show full dataset", value=False)

filtered_df = df.copy()

if "RainTomorrow" in filtered_df.columns:
    rain_filter = st.sidebar.selectbox(
        "🌧️ Filter by RainTomorrow",
        options=["All", 0, 1]
    )
    if rain_filter != "All":
        filtered_df = filtered_df[filtered_df["RainTomorrow"] == rain_filter]

numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if "MaxTemp" in filtered_df.columns:
    temp_range = st.sidebar.slider(
        "🌡️ MaxTemp Range",
        float(filtered_df["MaxTemp"].min()),
        float(filtered_df["MaxTemp"].max()),
        (
            float(filtered_df["MaxTemp"].min()),
            float(filtered_df["MaxTemp"].max())
        )
    )
    filtered_df = filtered_df[
        (filtered_df["MaxTemp"] >= temp_range[0]) &
        (filtered_df["MaxTemp"] <= temp_range[1])
    ]

# ---------------------------------
# Header
# ---------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌦️ Weather Forecasting Dashboard</h1>
    <p>Historical Weather Analysis + Rain Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# KPI Cards
# ---------------------------------
avg_temp = f"{filtered_df['MaxTemp'].mean():.1f} °C" if "MaxTemp" in filtered_df.columns else "N/A"
avg_humidity = f"{filtered_df['Humidity3pm'].mean():.1f} %" if "Humidity3pm" in filtered_df.columns else "N/A"
avg_pressure = f"{filtered_df['Pressure3pm'].mean():.1f}" if "Pressure3pm" in filtered_df.columns else "N/A"
rain_pct = f"{filtered_df['RainTomorrow'].mean() * 100:.1f} %" if "RainTomorrow" in filtered_df.columns else "N/A"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🌡️ Average Temperature</div>
        <div class="kpi-value">{avg_temp}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💧 Average Humidity</div>
        <div class="kpi-value">{avg_humidity}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🌬️ Average Pressure</div>
        <div class="kpi-value">{avg_pressure}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🌧️ Rain Probability</div>
        <div class="kpi-value">{rain_pct}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# Alerts
# ---------------------------------
with st.container():
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("🚨 Weather Alerts")

    alert_found = False

    if "MaxTemp" in filtered_df.columns and filtered_df["MaxTemp"].mean() > 35:
        st.warning("High Temperature Alert 🔥")
        alert_found = True

    if "Humidity3pm" in filtered_df.columns and filtered_df["Humidity3pm"].mean() > 80:
        st.warning("High Humidity Alert 💧")
        alert_found = True

    if not alert_found:
        st.success("No major weather alerts based on current filtered data ✅")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Analysis",
    "🤖 Prediction",
    "🗂️ Dataset",
    "💡 Insights"
])

# ---------------------------------
# TAB 1 - Analysis
# ---------------------------------
with tab1:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("📈 Weather Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        if "MaxTemp" in filtered_df.columns:
            fig1 = px.histogram(
                filtered_df,
                x="MaxTemp",
                nbins=30,
                title="Temperature Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True, key="hist_max_temp")

    with col_b:
        if "Humidity3pm" in filtered_df.columns and "MaxTemp" in filtered_df.columns:
            fig2 = px.scatter(
                filtered_df,
                x="Humidity3pm",
                y="MaxTemp",
                color="RainTomorrow" if "RainTomorrow" in filtered_df.columns else None,
                title="Humidity vs Temperature"
            )
            st.plotly_chart(fig2, use_container_width=True, key="scatter_humidity_temp")

    if "RainTomorrow" in filtered_df.columns:
        rain_counts = filtered_df["RainTomorrow"].value_counts().reset_index()
        rain_counts.columns = ["RainTomorrow", "Count"]

        fig3 = px.bar(
            rain_counts,
            x="RainTomorrow",
            y="Count",
            title="Rain Tomorrow Distribution"
        )
        st.plotly_chart(fig3, use_container_width=True, key="rain_distribution")

    st.subheader("🧠 Feature Importance")

    if hasattr(model, "feature_importances_"):
        try:
            feat_df = pd.DataFrame({
                "Feature": feature_columns_saved,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig4 = px.bar(
                feat_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Important Features"
            )
            st.plotly_chart(fig4, use_container_width=True, key="feature_importance")
        except Exception as e:
            st.info(f"Feature importance could not be displayed: {e}")
    else:
        st.info("This model does not provide feature importance.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# TAB 2 - Prediction
# ---------------------------------
with tab2:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("🤖 Rain Prediction")

    if "RainTomorrow" in filtered_df.columns and len(filtered_df) > 0:
        prediction_features = [col for col in filtered_df.columns if col != "RainTomorrow"]

        selected_row = st.selectbox(
            "Select a row index for prediction",
            options=filtered_df.index.tolist()
        )

        input_data = filtered_df.loc[[selected_row], prediction_features]
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("Prediction Result: Rain Expected Tomorrow ☔")
        else:
            st.success("Prediction Result: No Rain Expected Tomorrow ☀️")
    else:
        st.warning("RainTomorrow column not found or filtered data is empty.")

    st.markdown("---")
    st.subheader("🔮 Custom Rain Prediction")

    p1, p2, p3 = st.columns(3)

    humidity = p1.number_input("💧 Humidity3pm", min_value=0.0, max_value=100.0, value=80.0)
    pressure = p2.number_input("🌡️ Pressure3pm", min_value=900.0, max_value=1100.0, value=1008.0)
    windspeed = p3.number_input("🌬️ WindSpeed3pm", min_value=0.0, max_value=150.0, value=25.0)

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

            pred = model.predict(input_df)[0]

            if pred == 1:
                st.error("Model Result: Rain Expected Tomorrow ☔")
            else:
                st.success("Model Result: No Rain Expected Tomorrow ☀️")

            with st.expander("See Input Summary"):
                st.write({
                    "Humidity3pm": humidity,
                    "Pressure3pm": pressure,
                    "WindSpeed3pm": windspeed
                })

            with st.expander("Dataset-Based Logic"):
                if humidity >= 80 and pressure <= 1008 and windspeed >= 25:
                    st.warning("Higher chance of rain 🌧️")
                elif humidity >= 70 and pressure <= 1012 and windspeed >= 15:
                    st.info("Moderate chance of rain ⛅")
                else:
                    st.success("Lower chance of rain ☀️")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# TAB 3 - Dataset
# ---------------------------------
with tab3:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("🗂️ Dataset Preview")

    preview_df = filtered_df if show_full_data else filtered_df.head(20)
    st.dataframe(preview_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv,
        file_name="filtered_weather_data.csv",
        mime="text/csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# TAB 4 - Insights
# ---------------------------------
with tab4:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
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
        for item in insights:
            st.markdown(f"- {item}")
    else:
        st.info("No insights available from current dataset.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""
<div class="footer-text">
Built with ❤️ using Streamlit | Weather Forecasting & Analytics Dashboard
</div>
""", unsafe_allow_html=True)
