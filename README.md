# 🌦️ Weather Forecasting & Rain Prediction System

A complete **Data Analytics + Machine Learning project** that analyzes historical weather data and predicts whether it will rain tomorrow.

---

## 🚀 Project Overview

This project combines:

* 📊 Data Analysis (EDA)
* 🤖 Machine Learning
* 📈 Interactive Dashboard (Streamlit)

It provides insights into weather patterns and allows users to predict rainfall based on custom inputs.

---

## 🎯 Objectives

* Analyze historical weather data
* Identify patterns affecting rainfall
* Build a machine learning model to predict rain
* Create an interactive dashboard for visualization & prediction

---

## 🧠 Machine Learning Model

* Model Used: **Random Forest Classifier**
* Handles class imbalance using `class_weight='balanced'`
* Trained on processed weather dataset

---

## 📊 Features

### 🔹 Data Analysis

* Temperature distribution
* Humidity vs Temperature relationship
* Correlation heatmap

### 🔹 Dashboard (Streamlit)

* KPI Metrics (Temp, Humidity, Pressure, Wind)
* Weather Alerts 🚨
* Interactive Charts 📈
* Rain Distribution Visualization

### 🔹 Predictions

* Row-based prediction
* Custom user input prediction
* Dataset-based weather logic

---

## 🧾 Project Structure

```
weather_forecasting_project/
│
├── data/
│   └── processed_weather.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── weather_model.pkl
├── model_features.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/weather-forecasting-project.git
cd weather-forecasting-project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 🧪 Example Inputs (Custom Prediction)

Try these values to test rain prediction:

| Feature      | Value |
| ------------ | ----- |
| Humidity3pm  | 90    |
| Pressure3pm  | 1000  |
| WindSpeed3pm | 35    |

---

## 📌 Key Insights

* High humidity increases chances of rainfall
* Low pressure is strongly associated with rain
* Wind speed contributes to weather changes
* Dataset shows class imbalance, handled using model tuning

---

## 💼 Skills Demonstrated

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning (Classification)
* Model Evaluation
* Data Visualization
* Streamlit App Development

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Plotly
* Streamlit
* Joblib

---

## 📈 Future Improvements

* Add real-time weather API integration 🌐
* Improve model accuracy using advanced techniques
* Deploy on cloud (Streamlit Cloud / Render / AWS)
* Add probability-based predictions

---

## 🙌 Author

**Ashish (Data Analyst | ML Enthusiast)**

---

⭐ If you like this project, don't forget to star the repo!
