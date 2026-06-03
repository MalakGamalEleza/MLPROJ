import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: #0f0f14;
        color: #e8e6e1;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    /* Hero section */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    .hero-accent {
        color: #e05c5c;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8a8894;
        font-weight: 300;
        max-width: 520px;
        line-height: 1.6;
        margin-bottom: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(224, 92, 92, 0.12);
        border: 1px solid rgba(224, 92, 92, 0.3);
        color: #e05c5c;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.3rem 0.8rem;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }

    /* Section headers */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #e05c5c;
        margin-bottom: 1rem;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(224, 92, 92, 0.2);
    }

    /* Card containers */
    .card {
        background: #17171f;
        border: 1px solid #2a2a35;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* Override streamlit widget styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider {
        background: #1e1e28 !important;
    }

    div[data-testid="stSelectbox"] > div {
        background: #1e1e28;
        border: 1px solid #2a2a35;
        border-radius: 8px;
        color: #e8e6e1;
    }

    div[data-testid="stNumberInput"] input {
        background: #1e1e28;
        border: 1px solid #2a2a35;
        border-radius: 8px;
        color: #e8e6e1;
    }

    /* Slider accent color */
    .stSlider [data-testid="stTickBar"] { display: none; }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #e05c5c, #e05c5c) !important;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #c4c2be !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e05c5c 0%, #c23b3b 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 2rem;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ec6e6e 0%, #d14848 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(224, 92, 92, 0.35);
    }

    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, rgba(224,92,92,0.15), rgba(180,40,40,0.1));
        border: 1px solid rgba(224, 92, 92, 0.5);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(80,200,120,0.12), rgba(50,150,80,0.08));
        border: 1px solid rgba(80, 200, 120, 0.4);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
    }
    .result-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .result-high .result-title { color: #e05c5c; }
    .result-low .result-title { color: #50c878; }

    .result-prob {
        font-size: 3.5rem;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .result-high .result-prob { color: #e05c5c; }
    .result-low .result-prob { color: #50c878; }

    .result-label {
        font-size: 0.8rem;
        color: #8a8894;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* Risk factor pills */
    .risk-pill {
        display: inline-block;
        background: rgba(224, 92, 92, 0.12);
        border: 1px solid rgba(224, 92, 92, 0.25);
        color: #e05c5c;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        margin: 0.2rem;
    }

    /* Divider */
    hr { border-color: #2a2a35 !important; margin: 2rem 0 !important; }

    /* Info callout */
    .info-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #e05c5c;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #8a8894;
        line-height: 1.6;
        margin-top: 1.5rem;
    }

    /* Dataset stats strip */
    .stat-strip {
        background: #17171f;
        border: 1px solid #2a2a35;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        display: flex;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .stat-item { text-align: center; }
    .stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #e05c5c;
    }
    .stat-desc { font-size: 0.72rem; color: #8a8894; text-transform: uppercase; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)


# =====================================
# LOAD MODEL & SCALER
# =====================================

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("heart_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, True
    except Exception:
        return None, None, False

model, scaler, model_loaded = load_artifacts()


# =====================================
# HERO HEADER
# =====================================

col_hero, col_stat = st.columns([2, 1], gap="large")

with col_hero:
    st.markdown('<div class="hero-badge">🫀 BRFSS 2015 · CDC Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Heart Disease<br><span class="hero-accent">Risk Assessment</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Answer 21 clinical questions drawn from the CDC\'s Behavioral Risk Factor Surveillance System to estimate your cardiovascular risk profile.</p>', unsafe_allow_html=True)

with col_stat:
    st.markdown("""
    <div style="padding-top:1.2rem">
        <div class="card" style="margin-bottom:0.6rem">
            <div class="stat-num">253,680</div>
            <div class="stat-desc">Survey respondents</div>
        </div>
        <div class="card" style="margin-bottom:0.6rem">
            <div class="stat-num">9.4%</div>
            <div class="stat-desc">Prevalence in dataset</div>
        </div>
        <div class="card">
            <div class="stat-num">21</div>
            <div class="stat-desc">Health indicators</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ **Model files not found.** Place `heart_model.pkl` and `scaler.pkl` in the same directory. The form below is ready for when models are available.", icon="⚠️")

st.markdown("---")

# =====================================
# FORM — 3 SECTIONS
# =====================================

# ---- SECTION 1: Clinical Biomarkers ----
st.markdown('<div class="section-label">01 — Clinical Biomarkers</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    HighBP = st.selectbox("High Blood Pressure", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Have you ever been told you have high blood pressure?")

with c2:
    HighChol = st.selectbox("High Cholesterol", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Have you ever been told your cholesterol is high?")

with c3:
    CholCheck = st.selectbox("Cholesterol Check (5 yrs)", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Have you had a cholesterol check in the past 5 years?")

c4, c5 = st.columns([1, 2])

with c4:
    BMI = st.number_input("BMI", min_value=10.0, max_value=98.0, value=25.0, step=0.5,
        help="Body Mass Index. Normal range: 18.5–24.9")

with c5:
    Diabetes = st.selectbox(
        "Diabetes Status",
        [0, 1, 2],
        format_func=lambda x: {0: "No diabetes", 1: "Pre-diabetes / borderline", 2: "Diabetes (diagnosed)"}[x]
    )

# ---- SECTION 2: Health History ----
st.markdown('<div class="section-label">02 — Health History & Conditions</div>', unsafe_allow_html=True)

c6, c7, c8, c9 = st.columns(4)

with c6:
    Smoker = st.selectbox("Smoker", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Have you smoked at least 100 cigarettes in your entire life?")

with c7:
    Stroke = st.selectbox("Stroke History", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Have you ever been told you had a stroke?")

with c8:
    DiffWalk = st.selectbox("Difficulty Walking", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Do you have serious difficulty walking or climbing stairs?")

with c9:
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol Use", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Adult men ≥14 drinks/week, adult women ≥7 drinks/week")

# ---- SECTION 3: Lifestyle & Wellbeing ----
st.markdown('<div class="section-label">03 — Lifestyle & Wellbeing</div>', unsafe_allow_html=True)

c10, c11, c12 = st.columns(3)

with c10:
    PhysActivity = st.selectbox("Physical Activity", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Physical activity or exercise in past 30 days (outside of job)?")

with c11:
    Fruits = st.selectbox("Fruit Consumption", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Do you consume fruit 1 or more times per day?")

with c12:
    Veggies = st.selectbox("Vegetable Consumption", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes ✓",
        help="Do you consume vegetables 1 or more times per day?")

c13, c14 = st.columns(2)

with c13:
    MentHlth = st.slider(
        "Poor Mental Health Days (last 30 days)", 0, 30, 0,
        help="Days when mental health was not good (stress, depression, emotional problems)"
    )

with c14:
    PhysHlth = st.slider(
        "Poor Physical Health Days (last 30 days)", 0, 30, 0,
        help="Days when physical health was not good (illness, injury)"
    )

# ---- SECTION 4: Demographics & Access ----
st.markdown('<div class="section-label">04 — Demographics & Healthcare Access</div>', unsafe_allow_html=True)

c15, c16, c17 = st.columns(3)

with c15:
    Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

with c16:
    Age = st.select_slider(
        "Age Group",
        options=list(range(1, 14)),
        value=7,
        format_func=lambda x: {
            1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
            5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
            9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
        }[x]
    )

with c17:
    GenHlth = st.select_slider(
        "General Health",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x]
    )

c18, c19, c20 = st.columns(3)

with c18:
    Education = st.select_slider(
        "Education Level",
        options=list(range(1, 7)),
        value=5,
        format_func=lambda x: {
            1: "No school / K only", 2: "Grades 1–8",
            3: "Grades 9–11", 4: "Grade 12 / GED",
            5: "College 1–3 years", 6: "College 4+ years"
        }[x]
    )

with c19:
    Income = st.select_slider(
        "Household Income",
        options=list(range(1, 9)),
        value=6,
        format_func=lambda x: {
            1: "< $10K", 2: "$10K–$15K", 3: "$15K–$20K",
            4: "$20K–$25K", 5: "$25K–$35K", 6: "$35K–$50K",
            7: "$50K–$75K", 8: "≥ $75K"
        }[x]
    )

with c20:
    AnyHealthcare = st.selectbox("Health Insurance / Coverage", [0, 1],
        format_func=lambda x: "No coverage" if x == 0 else "Has coverage ✓")

NoDocbcCost = st.selectbox(
    "Skipped doctor visit due to cost (last 12 months)?", [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes — cost was a barrier"
)

# =====================================
# PREDICT BUTTON
# =====================================

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🫀 Assess My Heart Disease Risk", use_container_width=True)

# =====================================
# PREDICTION LOGIC
# =====================================

if predict_btn:
    input_data = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker,
        Stroke, Diabetes, PhysActivity, Fruits, Veggies,
        HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
        MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    ]], dtype=float)

    # Identify active risk factors
    risk_factors = []
    if HighBP == 1:         risk_factors.append("High Blood Pressure")
    if HighChol == 1:       risk_factors.append("High Cholesterol")
    if Smoker == 1:         risk_factors.append("Smoker")
    if Stroke == 1:         risk_factors.append("Stroke History")
    if Diabetes >= 1:       risk_factors.append("Diabetes / Pre-diabetes")
    if DiffWalk == 1:       risk_factors.append("Difficulty Walking")
    if HvyAlcoholConsump == 1: risk_factors.append("Heavy Alcohol Use")
    if PhysActivity == 0:   risk_factors.append("No Physical Activity")
    if GenHlth >= 4:        risk_factors.append("Poor General Health")
    if BMI >= 30:           risk_factors.append(f"BMI {BMI:.0f} (Obese)")
    if PhysHlth >= 14:      risk_factors.append(f"{PhysHlth} Poor Physical Days")
    if MentHlth >= 14:      risk_factors.append(f"{MentHlth} Poor Mental Days")
    if Age >= 9:            risk_factors.append("Age 60+")

    st.markdown("---")
    st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)

    if model_loaded:
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            res_col, factor_col = st.columns([1, 1], gap="large")

            with res_col:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-high">
                        <div class="result-title">⚠️ Elevated Risk Detected</div>
                        <div class="result-prob">{probability:.0%}</div>
                        <div class="result-label">Predicted risk probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low">
                        <div class="result-title">✅ Low Risk Profile</div>
                        <div class="result-prob">{probability:.0%}</div>
                        <div class="result-label">Predicted risk probability</div>
                    </div>
                    """, unsafe_allow_html=True)

            with factor_col:
                st.markdown("**Active Risk Factors Detected**")
                if risk_factors:
                    pills_html = "".join(f'<span class="risk-pill">⚡ {rf}</span>' for rf in risk_factors)
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#50c878; font-size:0.9rem">✅ No major risk factors identified</span>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#c4c2be">About this result:</strong><br>
                    This prediction is based on a machine learning model trained on CDC BRFSS 2015 data
                    ({risk_factors.__len__()} risk factor(s) active). This tool is <em>not</em> a medical
                    diagnosis. Please consult a qualified healthcare professional for clinical evaluation.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        # Fallback: show risk factor summary without model
        res_col, factor_col = st.columns([1, 1], gap="large")
        risk_score = len(risk_factors)

        with res_col:
            if risk_score >= 4:
                level, color, emoji = "Elevated Risk (Rule-Based)", "#e05c5c", "⚠️"
                css_class = "result-high"
            else:
                level, color, emoji = "Moderate / Low Risk", "#50c878", "✅"
                css_class = "result-low"

            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-title">{emoji} {level}</div>
                <div class="result-prob">{risk_score}</div>
                <div class="result-label">Risk factors identified (no model loaded)</div>
            </div>
            """, unsafe_allow_html=True)

        with factor_col:
            st.markdown("**Active Risk Factors**")
            if risk_factors:
                pills_html = "".join(f'<span class="risk-pill">⚡ {rf}</span>' for rf in risk_factors)
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#50c878">✅ No major risk factors identified</span>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                ℹ️ Model files (<code>heart_model.pkl</code>, <code>scaler.pkl</code>) were not found.
                Showing a simplified rule-based summary instead.
                Train and save your model to enable full ML predictions.
            </div>
            """, unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================

st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#3a3a48; font-size:0.78rem; letter-spacing:0.05em">
    CDC BRFSS 2015 Dataset · For Educational & Research Use Only · Not a Medical Device
</p>
""", unsafe_allow_html=True)