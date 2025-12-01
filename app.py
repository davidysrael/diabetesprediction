import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import base64

# Model core
model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

# Background
def load_bg(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

BG_PATH = r"main/Welcome to BloodBeaconPH.png"
bg_base64 = load_bg(BG_PATH)

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
    .risk-btn-wrap {{
      display:flex;
      justify-content:center;
      margin:1.2rem 0;
    }}
  </style>
  """,
  unsafe_allow_html=True,
)

# Risk heuristic
def risk_likelihood(a, g, h, b):
  s = 0
  s += 1.5 if (a > 45) else 0
  s += 2 if (a > 60) else 0
  s += 2.5 if (g > 140) else 0
  s += 3 if (g > 200) else 0
  s += 2 if (h > 5.7) else 0
  s += 3 if (h > 6.5) else 0
  s += 1.5 if (b > 27) else 0
  s += 2.5 if (b > 30) else 0
  return min(s / 12, 1.0)

# ---- Modal control ----
if "show_modal" not in st.session_state:
  st.session_state.show_modal = False

def open_predictor():
  st.session_state.show_modal = True

# ---- Landing UI ----
st.title("🩸 BloodBeaconPH")
st.write("Dr. Gary Glucose A.I active. PH biomarker systems operational.")

st.markdown('<div class="risk-btn-wrap">', unsafe_allow_html=True)
st.button("🧪 Predict Your Risk of Diabetes", on_click=open_predictor)
st.markdown("</div>", unsafe_allow_html=True)

st.subheader("📊 Patient Demographics")
chart_key = st.radio(
  "",
  list(demo_paths := {
    "Age Distribution": "main/age_distribution.png",
    "Blood Sugar": "main/blood_sugar_distribution.png",
    "BMI Profile": "main/bmi_distribution.png",
    "Gender": "main/gender_distribution.png",
    "HbA1c": "main/hba1c_distribution.png",
    "Heart Disease": "main/heart_disease_distribution.png",
    "Hypertension": "main/hypertension_distribution.png",
  }).keys(),
  horizontal=True,
  key="cohort_radio",
)
st.image(demo_paths[chart_key], use_column_width=True)

# ---- Render modal only after rerun ----
if st.session_state.show_modal:
  with st.modal("🔍 Diabetes Risk Scanner"):

    # Predictor console UI
    st.subheader("Clinical Inputs")
    gender = st.selectbox("Gender", ["Male","Female"])
    age = st.number_input("Age (years)", 10, 120, 30)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
    glucose = st.number_input("Glucose (mg/dL)", 70, 400, 100)

    # BMI section (no delay + instant rerun)
    st.write("")
    w = st.number_input("Weight (kg)", 1.0, 300.0, 70.0)
    h = st.number_input("Height (cm)", 30.0, 250.0, 170.0)
    if st.button("Calculate BMI"):
      bmi_val = round(w / ((h/100) ** 2), 2)
      st.session_state.bmi_calc_value = bmi_val
      st.experimental_rerun()  # forces immediate reflection

    bmi = st.session_state.get("bmi_calc_value")
    if bmi:
      st.metric("BMI", bmi)

    # Glossary relocated here
    with st.expander("🧾 Medical Glossary"):
      st.write("""
      HbA1c — avg glucose over 2–3 months  
      BMI — height/weight indexed  
      Hypertension — BP risk factor
      """)

    # Run analysis
    if (st.button("📡 Analyze Risk", disabled=(bmi is None))):
      arr = np.array([[1 if gender=="Male" else 0, age, hypertension, heart_disease, bmi, hba1c, glucose]])
      arr_s = scaler.transform(arr)
      pred = model.predict(arr_s)[0]
      r_val = risk_likelihood(age, glucose, hba1c, bmi)

      # Chart
      st.subheader("Biomarker Contribution")
      vals = [age/120*100, bmi/50*100, glucose/400*100, hba1c/15*100]
      labs = ["Age","BMI","Glucose","HbA1c"]
      f, a = plt.subplots()
      a.bar(labs, vals)
      for i, v in enumerate(vals):
        a.text(i, v+2, f"{v:.1f}%", ha="center", weight="bold")
      st.pyplot(f)

      # Live radar
      st.progress(r_val)
      st.caption(f"Threat Index: {r_val*100:.1f}%")

      if pred == 1:
        st.error("🚨 High risk detected")
      else:
        st.success("✅ Risk nominal")
        st.balloons()

    # Close modal cleanly
    if st.button("Close Scanner"):
      st.session_state.show_modal = False
      st.experimental_rerun()
