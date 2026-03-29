import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Weather AI Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------
# GLASS UI CSS 🔥
# -------------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
}

/* Glass Cards */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

/* Header */
.header {
    text-align: center;
    padding: 20px;
}
.header h1 {
    font-size: 3rem;
    color: #fff;
}
.header p {
    font-size: 1.2rem;
    color: #ddd;
}

/* KPI */
.kpi {
    text-align: center;
}
.kpi h2 {
    margin: 0;
    font-size: 2rem;
}
.kpi p {
    margin: 0;
    font-size: 0.9rem;
}

/* Footer */
.footer {
    text-align:center;
    color:#ccc;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_weather_eda.csv"
MODEL_PATH = BASE_DIR / "weather_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

# -------------------------
# LOAD
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)

df = load_data()
model = load_model()
features = load_features()

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="header">
<h1>🌦️ Weather AI Dashboard</h1>
<p>Advanced Analytics + AI Rain Prediction</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# KPI SECTION
# -------------------------
c1,c2,c3,c4 = st.columns(4)

def card(title, value):
    return f"""
    <div class="glass kpi">
        <h2>{value}</h2>
        <p>{title}</p>
    </div>
    """

c1.markdown(card("🌡️ Avg Temp", f"{df['MaxTemp'].mean():.1f}°C"), unsafe_allow_html=True)
c2.markdown(card("💧 Humidity", f"{df['Humidity3pm'].mean():.1f}%"), unsafe_allow_html=True)
c3.markdown(card("🌬️ Wind", f"{df['WindSpeed3pm'].mean():.1f}"), unsafe_allow_html=True)
c4.markdown(card("🌧️ Rain %", f"{df['RainTomorrow'].mean()*100:.1f}%"), unsafe_allow_html=True)

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🤖 Prediction", "📁 Data"])

# -------------------------
# ANALYSIS TAB
# -------------------------
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    fig1 = px.histogram(df, x="MaxTemp", title="Temperature Distribution")
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Humidity3pm", y="MaxTemp", color="RainTomorrow")
    col2.plotly_chart(fig2, use_container_width=True)

    if hasattr(model,"feature_importances_"):
        imp = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig3 = px.bar(imp.head(10), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PREDICTION TAB
# -------------------------
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    h = col1.number_input("Humidity",0.0,100.0,80.0)
    p = col2.number_input("Pressure",900.0,1100.0,1008.0)
    w = col3.number_input("Wind",0.0,150.0,25.0)

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame(columns=features)
        input_df.loc[0]=0

        input_df.at[0,"Humidity3pm"]=h
        input_df.at[0,"Pressure3pm"]=p
        input_df.at[0,"WindSpeed3pm"]=w

        pred = model.predict(input_df)[0]

        if pred==1:
            st.error("🌧️ Rain Expected")
        else:
            st.success("☀️ No Rain")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# DATA TAB
# -------------------------
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.dataframe(df.head(50))
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<div class="footer">
Built by Ashish 🚀 | AI Weather Dashboard
</div>
""", unsafe_allow_html=True)
