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

BG_PATH = r"main2/Welcome to BloodBeaconPH.png"
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


# =========================================================
# LOAD NEW MODEL
# =========================================================
model = joblib.load("gbdt_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")  # keep scaler unchanged



# =========================================================
# PATIENT PROFILE — UPDATED INPUTS
# =========================================================
st.title("🩸 BloodBeaconPH")
st.write("Dr. Gary Glucose at your service. I am a Machine Learning powered system for predicting your risk of diabetes configured for PH Clinical trends.")

with st.expander("🧾 PH Medical Glossary"):
  st.write("""
  DPF — Diabetes Pedigree Function  
  Glucose mg/dL — sugar level  
  BloodPressure — arterial pressure  
  SkinThickness — triceps fold thickness  
  Insulin — fasting insulin  
  BMI — weight-to-height ratio  
  """)


# -------------------------------
# PATIENT PROFILE (NEW)
# -------------------------------
st.subheader("🧍 Patient Profile")

age = st.text_input("Age (years)   [max: 80]", value=("30"))
try:
  age = int(float(age))
except:
  age = 30


# -------------------------------
# PATIENT BIOMARKERS (NEW INPUTS)
# -------------------------------
st.subheader("🧬 Patient Biomarkers")

pregnancies = st.text_input("Pregnancies   [max: 17]", value=("0"))
glucose = st.text_input("Glucose (mg/dL)   [max: 200]", value=("100.00"))
bp = st.text_input("BloodPressure (mmHg)   [max: 122]", value=("70.00"))
skin = st.text_input("SkinThickness (mm)   [max: 99]", value=("20.00"))
insulin = st.text_input("Insulin (µU/mL)   [max: 845]", value=("79"))
dpf = st.text_input("DPF (0.00 format)   [max: 2.42]", value=("0.50"))

# Parse safely
try: pregnancies = int(float(pregnancies))
except: pregnancies = 0

try: glucose = round(float(glucose), 2)
except: glucose = 100.00

try: bp = round(float(bp), 2)
except: bp = 70.00

try: skin = round(float(skin), 2)
except: skin = 20.00

try: insulin = round(float(insulin), 2)
except: insulin = 79.00

try: dpf = round(float(dpf), 2)
except: dpf = 0.50



# =========================================================
# BMI CALCULATOR (UNCHANGED)
# =========================================================
st.subheader("📏 BMI Calculator")

if ("bmi_calc_value" not in st.session_state):
  st.session_state.bmi_calc_value = None

weight = st.text_input("Weight (kg)", value=("70.00"))
height = st.text_input("Height (cm)", value=("170.00"))

try: weight = float(weight)
except: weight = 70.00

try: height = float(height)
except: height = 170.00

if (st.button("Compute BMI", key=("btn_bmi"))):
  bmi_temp = weight / ((height / 100) ** 2)
  st.session_state.bmi_calc_value = round(bmi_temp, 2)
  st.metric("✅ Calculated BMI", f"{bmi_temp:.2f}")

bmi = st.session_state.bmi_calc_value



# =========================================================
# NEW METRICS (Age, BMI, Glucose, DPF)
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", f"{age}")
c2.metric("BMI", ("--" if (bmi is None) else f"{bmi:.2f}"))
c3.metric("Glucose", f"{glucose:.2f}")
c4.metric("DPF", f"{dpf:.2f}")

scan_ready = bmi is not None
console = st.empty()



# =========================================================
# INITIATE SCAN — NEW RISK SYSTEM
# =========================================================
if (st.button("🔍 Initiate Beacon Scan", key=("btn_predict"), disabled=(not scan_ready))):

  st.subheader("📊 Biomarker Breakdown (PH Risk %)")

  # Convert into % risk relative to maximum allowed values
  values = [
    (age / 80) * 100,
    (pregnancies / 17) * 100,
    (glucose / 200) * 100,
    (bp / 122) * 100,
    (skin / 99) * 100,
    (insulin / 845) * 100,
    (dpf / 2.42) * 100,
    (bmi / 60) * 100  # BMI typically up to 60 clinically
  ]

  labels = [
    "Age",
    "Pregnancies",
    "Glucose",
    "BP",
    "Skin",
    "Insulin",
    "DPF",
    "BMI"
  ]

  def bar_color(v):
    if (v >= 90): return "red"
    if (v >= 80): return "orangered"
    if (v >= 70): return "darkorange"
    if (v >= 60): return "orange"
    return "gray"

  colors = list(map(bar_color, values))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.bar(labels, values)

  for i, bar in enumerate(ax.patches):
    bar.set_facecolor(colors[i])

  ax.set_title("PH Clinical Biomarker Levels", fontsize=14)
  ax.set_ylabel("Risk Contribution (%)", fontsize=12)
  ax.set_ylim(0, 110)
  ax.grid(axis=("y"), alpha=0.2)

  for i, v in enumerate(values):
    ax.text(i, v + 2, f"{v:.1f}%", ha=("center"), fontsize=12, weight=("bold"))

  plt.tight_layout()
  st.pyplot(fig)

  # Live Radar Score
  r_live = np.mean(values) / 100
  st.subheader("📡 Live Risk Radar (PH Clinical Index)")
  st.progress(r_live)
  st.caption(f"Current system threat index: {r_live * 100:.1f}%")

  console.write("Calibrating hematology sensors...")
  console.write("Reading glucose and biomarker matrix...")
  console.write("Firing predictive core...")

  # NEW ML INPUT ORDER
  X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
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
