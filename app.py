import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import base64

st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

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
    input[type="number"] {{
      all: unset !important;
      display: block;
      width: 100%;
      background-color: white;
      color: black;
      padding: 0.4rem;
      border-radius: 0.6rem;
      border: 1px solid #ccc;
      font-size: 1rem;
    }}

    
    div.stButton > button#btn_predict {{
      background: linear-gradient(90deg, orangered, darkorange);
      color: white;
      padding: 0.6rem 1.4rem;
      border-radius: 10px;
      border: none;
      font-size: 16px;
      font-weight: 600;
      transition: 0.25s ease-in-out;
    }}
    div.stButton > button#btn_predict:hover {{
      background: linear-gradient(90deg, darkorange, orangered);
      transform: scale(1.03);
      cursor: pointer;
    }}
  </style>
  """,
  unsafe_allow_html=True,
)


st.markdown("""
<script>
document.addEventListener("input",(e)=>{
  const t=e.target;
  if(t.tagName==="INPUT" && t.type==="text"){

      // AGE strictly integer-only
      if(t.placeholder.includes("Age")){
          t.value = t.value.replace(/[^0-9]/g,"");
          return;
      }

      // Others: allow digits + single decimal
      t.value = t.value
        .replace(/[^0-9.]/g,"")
        .replace(/\\.(?=.*\\.)/g,"");
  }
});
</script>
""", unsafe_allow_html=True)


model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

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
  st.write("Dr. Gary Glucose A.I")
  st.caption("Endocrinologist • PH Biomarker Specialist")
  st.write("""
  Specialized in glucose pattern recognition and chronic illness risk assessment.  
  Passionate about diagnostics and preventive healthcare.
  """)


st.title("🩸 BloodBeaconPH")
st.write("Dr. Gary Glucose at your service. I am a Machine Learning powered system for predicting your risk of diabetes configured for PH Clinical trends.")

with st.expander("🧾 PH Medical Glossary"):
  st.write("""
  HbA1c — measures average blood sugar in the last 2–3 months.  
  Glucose mg/dL — current blood sugar concentration.  
  BMI — body mass index based on height and weight.  
  Hypertension — high blood pressure, a diabetes risk factor.
  """)


st.subheader("🧍 Patient Profile")

gender = st.selectbox("Gender", ["Male","Female"], key=("gender_select_main"))
age = st.text_input("Age (years)", value=("30"))

try:
  age = int(float(age))
except:
  age = 30


st.subheader("📏 BMI Calculator")

if ("bmi_calc_value" not in st.session_state):
  st.session_state.bmi_calc_value = None

weight = st.text_input("Weight (kg)", value=("70.00"))
height = st.text_input("Height (cm)", value=("170.00"))

try:
  weight = round(float(weight), 2)
except:
  weight = 70.00

try:
  height = round(float(height), 2)
except:
  height = 170.00

if (st.button("Compute BMI", key=("btn_bmi"))):
  bmi_temp = weight / ((height / 100) ** 2)
  st.session_state.bmi_calc_value = round(bmi_temp, 2)
  st.metric("✅ Calculated BMI", f"{bmi_temp:.2f}")

bmi = st.session_state.bmi_calc_value

st.subheader("🧬 Patient Biomarkers")

hba1c = st.text_input("HbA1c (%)", value=("5.50"))
glucose = st.text_input("Blood Glucose (mg/dL)", value=("100.00"))

hypertension = st.selectbox("Hypertension [0=none, 1=yes]", [0, 1], key=("input_htn_main"))
heart_disease = st.selectbox("Heart Disease [0=none, 1=yes]", [0, 1], key=("input_hd_main"))

try:
  hba1c = round(float(hba1c), 2)
except:
  hba1c = 5.50

try:
  glucose = round(float(glucose), 2)
except:
  glucose = 100.00

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", f"{age}")
c2.metric("BMI", ("--" if (bmi is None) else f"{bmi:.2f}"))
c3.metric("Glucose", f"{glucose:.2f}")
c4.metric("HbA1c", f"{hba1c:.2f}")

scan_ready = False
if (bmi is None):
  st.warning("🔐 Scan lock: BMI must be computed first.")
else:
  scan_ready = True

gender_encoded = 1 if (gender == "Male") else 0
X = np.array([[gender_encoded, age, hypertension, heart_disease, bmi, hba1c, glucose]])
console = st.empty()


if (st.button("🔍 Initiate Beacon Scan", key=("btn_predict"), disabled=(not scan_ready))):
  st.subheader("📊 Biomarker Breakdown FINDRISC (Finnish Diabetes Risk Score)")

 

  if age < 35: age_pts = 0
  elif age < 45: age_pts = 2
  elif age < 55: age_pts = 4
  else: age_pts = 6


  if bmi is None:
    bmi_pts = 0
  elif bmi < 23: bmi_pts = 0
  elif bmi < 25: bmi_pts = 1
  elif bmi < 30: bmi_pts = 2
  else: bmi_pts = 4

  
  htn_pts = 3 if hypertension == 1 else 0

  
  heart_pts = 3 if heart_disease == 1 else 0

  
  if glucose < 100: glu_pts = 0
  elif glucose < 126: glu_pts = 3
  else: glu_pts = 6

 
  if hba1c < 5.7: a1c_pts = 0
  elif hba1c < 6.5: a1c_pts = 3
  else: a1c_pts = 6

  
  values = [
    (age_pts / 6) * 100,
    (bmi_pts / 4) * 100,
    (glu_pts / 6) * 100,
    (a1c_pts / 6) * 100,
    (htn_pts / 3) * 100,
    (heart_pts / 3) * 100
  ]

  labels = [
    "Age",
    "BMI",
    "Glucose",
    "HbA1c",
    "Hypertension",
    "Heart Disease"
  ]

  def bar_color(v):
    if (v >= 90): return "red"
    if (v >= 80): return "orangered"
    if (v >= 70): return "darkorange"
    if (v >= 60): return "orange"
    return "gray"

  colors = list(map(bar_color, values))

 
  fig, ax = plt.subplots()
  ax.bar(labels, values)

  for i, bar in enumerate(ax.patches):
    bar.set_facecolor(colors[i])

  ax.set_title("PH Clinical Biomarker Levels - FINDRISC (Finnish Diabetes Risk Score)", fontsize=(14))
  ax.set_ylabel("Risk Contribution (%)", fontsize=(12))
  ax.set_ylim(0, 110)
  ax.grid(axis=("y"), alpha=0.2)

  for i, v in enumerate(values):
    ax.text(i, v + 2, f"{v:.1f}%", ha=("center"), fontsize=(12), weight=("bold"))

  st.pyplot(fig)


  total_findrisc = age_pts + bmi_pts + glu_pts + a1c_pts + htn_pts + heart_pts
  r_live = total_findrisc / 28

  st.subheader("📡 Live Risk Radar (PH Clinical Index)")
  st.progress(r_live)
  st.caption(f"Current system threat index: {r_live * 100:.1f}%")

  console.write("Calibrating hematology sensors...")
  console.write("Reading glucose and HbA1c matrix...")
  console.write("Firing predictive core...")

  X_scaled = scaler.transform(X)
  result = model.predict(X_scaled)[0]

  if (result == 1):
    st.error("🚨 High diabetes risk detected.")
    console.write("A probability of insulin resistance alert.")
  else:
    st.success("✅ No high diabetes risk detected.")
    st.balloons()
    console.write("All vitals optimal, sir.")

st.write("---")
st.caption("Diagnostics by Dr. Gary Glucose from BloodBeaconPH system core.")
