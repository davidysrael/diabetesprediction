import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
import pandas as pd
import plotly.express as px

# Load ML assets
model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Auto-refresh dataset on rerun
DATA_PATH = r"diabetes_prediction_dataset.csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Background auto-load from local `main` folder
def load_bg(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

BG_PATH = r"main/Welcome to BloodBeaconPH.png"
bg_base64 = load_bg(BG_PATH)

st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

# Apply UI background
st.markdown(
  f"""
  <style>
    .stApp {{
      background: url("data:image/png;base64,{bg_base64}");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      background-attachment: fixed;
    }}
    .block-container {{
      background-color: rgba(0,0,0,0.12);
      border-radius: 2rem;
      padding: 2rem;
      backdrop-filter: blur(6px);
    }}
  </style>
  """,
  unsafe_allow_html=True,
)

# Risk scoring function
def risk_likelihood(a, g, h, b):
  score = 0
  score += 1.5 if (a > 45) else 0
  score += 2 if (a > 60) else 0
  score += 2.5 if (g > 140) else 0
  score += 3 if (g > 200) else 0
  score += 2 if (h > 5.7) else 0
  score += 3 if (h > 6.5) else 0
  score += 1.5 if (b > 27) else 0
  score += 2.5 if (b > 30) else 0
  return min(score / 12, 1.0)

# Sidebar
with st.sidebar:
  st.subheader("🧬 System Physician Console")
  st.write("**AI Core Physician — BloodBeacon**")

# Header
st.title("🩸 BloodBeaconPH")
st.write("Online. Diagnostics primed.")

# Inputs
gender = st.selectbox("Gender", ["Male","Female"], key=("gender_select_main"))
age = st.number_input("Age (years)", min_value=(10), max_value=(80), value=(30), key=("age_input_main"))
hypertension = st.selectbox("Hypertension [0=none,1=yes]", [0,1], key=("input_htn_main"))
heart_disease = st.selectbox("Heart Disease [0=none,1=yes]", [0,1], key=("input_hd_main"))
hba1c = st.number_input("HbA1c (%)", min_value=(4.0), max_value=(9.0), value=(5.5), key=("input_hba1c_main"))
glucose = st.number_input("Blood Glucose (mg/dL)", min_value=(70), max_value=(300), value=(100), key=("input_glucose_main"))

bmi = st.session_state.bmi_calc_value if ("bmi_calc_value" in st.session_state) else None

# Live radar preview
r_live = 0 if (bmi is None) else risk_likelihood(age, glucose, hba1c, bmi)
st.subheader("📡 Live Clinical Risk Radar")
st.progress(r_live)
st.caption(f"Threat index nominal: {r_live * 100:.1f}%")

# C) New expander for dataset statistics, refreshed on rerun
with st.expander("📊 Dataset Feature Statistics"):
  RANGES = {
    "age": (0.08, 80.0),
    "bmi": (10.01, 95.69),
    "HbA1c_level": (3.5, 9.0),
    "blood_glucose_level": (80, 300),
    "hypertension": (0, 1),
    "heart_disease": (0, 1),
    "diabetes": (0, 1)
  }

  st.write("**Feature Ranges**")
  for f, (mn, mx) in RANGES.items():
    st.write(f"{f} → Min: {mn} | Max: {mx}")

  def render_hist(feature, bins):
    mn, mx = RANGES[feature]
    steps = np.linspace(mn, mx, bins + 1)
    fig = px.histogram(df, x=feature, bins=steps, range_x=[mn, mx])
    fig.update_traces(
      hovertemplate="Range: %{xbin.start:.2f} – %{xbin.end:.2f}<br>Count: %{y}<extra></extra>"
    )
    fig.update_layout(title=f"{feature} Distribution", xaxis_title="Range", yaxis_title="Count")
    st.plotly_chart(fig)

  def render_flag(feature):
    tally = df[feature].value_counts().sort_index()
    fig = px.bar(x=tally.index, y=tally.values)
    fig.update_traces(
      hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"
    )
    fig.update_layout(title=f"{feature} Distribution", xaxis_title=feature, yaxis_title="Count")
    st.plotly_chart(fig)

  render_hist("age", 20)
  render_hist("bmi", 18)
  render_hist("HbA1c_level", 15)
  render_hist("blood_glucose_level", 15)

  render_flag("hypertension")
  render_flag("heart_disease")
  render_flag("diabetes")

  gender_counts = df["gender"].value_counts()
  fig_gender = px.pie(names=gender_counts.index, values=gender_counts.values, hole=0.35, title="Gender Distribution")
  st.plotly_chart(fig_gender)

# Prediction scan button
if (st.button("🔍 Initiate Beacon Scan", key=("btn_predict"), disabled=(bmi is None))):
  console = st.empty()
  X_data = np.array([[1 if (gender == "Male") else 0, age, hypertension, heart_disease, 25.0 if (bmi is None) else bmi, hba1c, glucose]])
  X_scaled = scaler.transform(X_data)
  out = model.predict(X_scaled)[0]

  if (out == 1):
    st.error("🚨 Insulin resistance probability HIGH.")
    console.write("Alert. Recommend clinical follow-up.")
  else:
    st.success("✅ Biomarkers stable. Risk LOW.")
    st.balloons()
    console.write("All vitals aligned, sir.")
