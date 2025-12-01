import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import base64

model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

def load_bg(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

bg_base64 = load_bg("C:/Users/tolen/OneDrive/Documents/Diabetes Prediction/Welcome to BloodBeaconPH.png")

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

with st.sidebar:
  st.subheader("👨‍⚕️ Physician Console")
  st.write("**Dr. Harris A1C**")
  st.caption("Endocrinologist • PH Biomarker Specialist")
  st.write("""
  Specialized in glucose pattern recognition and chronic illness risk assessment.  
  Passionate about AI-assisted diagnostics and preventive healthcare.
  """)

st.title("🩸 BloodBeaconPH")
st.write("Hi, Dr. Gary Glucose online. I am an AI diabetes risk scanner tuned for PH clinical flow.")

with st.expander("🧾 PH Medical Glossary"):
  st.write("""
  **HbA1c** — measures average blood sugar in the last 2–3 months.  
  **Glucose mg/dL** — current blood sugar concentration.  
  **BMI** — body mass index based on height and weight.  
  **Hypertension** — high blood pressure, a diabetes risk factor.
  """)

# ----- INPUTS -----
gender = st.selectbox("Gender", ["Male","Female"], key=("gender_select_main"))
age = st.number_input("Age (years)", min_value=(10), max_value=(80), value=(30), key=("age_input_main"))
hypertension = st.selectbox("Hypertension [0=none, 1=yes]", [0,1], key=("input_htn_main"))
heart_disease = st.selectbox("Heart Disease [0=none, 1=yes]", [0,1], key=("input_hd_main"))
hba1c = st.number_input("HbA1c (%)", min_value=(4.0), max_value=(9.0), value=(5.5), key=("input_hba1c_main"))
glucose = st.number_input("Blood Glucose (mg/dL)", min_value=(70), max_value=(300), value=(100), key=("input_glucose_main"))

if ("bmi_calc_value" not in st.session_state):
  st.session_state.bmi_calc_value = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("BMI", ("--" if (st.session_state.bmi_calc_value is None) else st.session_state.bmi_calc_value))
c3.metric("Glucose", glucose)
c4.metric("HbA1c", hba1c)

with st.expander("📏 BMI Calculator"):
  weight = st.number_input("Weight (kg)", min_value=(1.0), max_value=(300.0), value=(70.0), key=("bmi_w"))
  height = st.number_input("Height (cm)", min_value=(30.0), max_value=(250.0), value=(170.0), key=("bmi_h"))
  if (st.button("Compute BMI", key=("btn_bmi"))):
    bmi_temp = weight / ((height / 100) ** 2)
    st.session_state.bmi_calc_value = round(bmi_temp, 2)
    st.metric("Calculated BMI", f"{bmi_temp:.2f}")

bmi = st.session_state.bmi_calc_value

# ----- VALIDATION INTERLOCK -----
if (bmi is None):
  st.warning("🔐 Scan lock: compute BMI first to proceed.")
  scan_ready = False
else:
  scan_ready = True

gender_encoded = 1 if (gender == "Male") else 0
X = np.array([[gender_encoded, age, hypertension, heart_disease, bmi, hba1c, glucose]])
console = st.empty()

if (st.button("🔍 Initiate Beacon Scan", key=("btn_predict"), disabled=(not scan_ready))):

  st.subheader("🧬 Biomarker Breakdown")
  values = [age/80 * 100, bmi/40 * 100, glucose/300 * 100, hba1c/9 * 100]
  labels = ["Age","BMI","Glucose","HbA1c"]

  # define colors by threshold for bars only
  def bar_color(v):
    if (v >= 90):
      return "red"
    if (v >= 80):
      return "orangered"
    if (v >= 70):
      return "darkorange"
    if (v >= 60):
      return "orange"
    return "gray"

  colors = list(map(bar_color, values))

  fig, ax = plt.subplots()
  ax.bar(labels, values, color=None)
  for i, bar in enumerate(ax.patches):
    bar.set_facecolor(colors[i])

  ax.set_title("PH Clinical Biomarker Levels", fontsize=(14))
  ax.set_ylabel("Risk Contribution (%)", fontsize=(12))
  ax.set_ylim(0, 110)
  ax.grid(axis=("y"), alpha=0.2)

  for i, v in enumerate(values):
    ax.text(i, v + 2, f"{v:.1f}%", ha=("center"), fontsize=(12), weight=("bold"))

  st.pyplot(fig)

  r_live = risk_likelihood(age, glucose, hba1c, bmi)
  st.subheader("📡 Live Risk Radar")
  st.progress(r_live)
  st.caption(f"Current system threat index: {r_live * 100:.1f}%")

  console.write("Calibrating hematology sensors...")
  console.write("Reading glucose and HbA1c matrix...")
  console.write("Firing predictive core...")

  X_scaled = scaler.transform(X)
  result = model.predict(X_scaled)[0]

  if (result == 1):
    st.error("🚨 High risk detected.")
    console.write("A probability of insulin resistance alert.")
  else:
    st.success("✅ No high risk detected.")
    st.balloons()
    console.write("You're in good shape.")

st.write("---")
st.caption("Diagnostics completed by Dr. Gary Glucose from BloodBeaconPH system core.")
