import streamlit as st
import numpy as np
import joblib

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0f0f14; color: #e8e6e1; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 720px; }
    .hero-badge {
        display: inline-block;
        background: rgba(224,92,92,0.12);
        border: 1px solid rgba(224,92,92,0.3);
        color: #e05c5c;
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.14em; text-transform: uppercase;
        padding: 0.28rem 0.8rem; border-radius: 100px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem; font-weight: 800;
        letter-spacing: -0.03em; line-height: 1.1;
        color: #fff; margin-bottom: 0.5rem;
    }
    .hero-accent { color: #e05c5c; }
    .hero-sub {
        font-size: 0.95rem; color: #8a8894;
        font-weight: 300; line-height: 1.65;
        margin-bottom: 0;
    }
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.62rem; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
        color: #e05c5c; margin-bottom: 0.9rem; margin-top: 2rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(224,92,92,0.2);
    }
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #c4c2be !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e05c5c 0%, #c23b3b 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.85rem 2rem;
        font-family: 'Syne', sans-serif;
        font-size: 1rem; font-weight: 700;
        letter-spacing: 0.04em; margin-top: 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ec6e6e 0%, #d14848 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(224,92,92,0.35);
    }
    .result-high {
        background: linear-gradient(135deg, rgba(224,92,92,0.15), rgba(180,40,40,0.1));
        border: 1px solid rgba(224,92,92,0.5);
        border-radius: 14px; padding: 2rem; text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(80,200,120,0.12), rgba(50,150,80,0.08));
        border: 1px solid rgba(80,200,120,0.4);
        border-radius: 14px; padding: 2rem; text-align: center;
    }
    .result-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; margin-bottom: 0.5rem; }
    .result-high .result-title { color: #e05c5c; }
    .result-low  .result-title { color: #50c878; }
    .result-prob { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800; letter-spacing: -0.04em; }
    .result-high .result-prob { color: #e05c5c; }
    .result-low  .result-prob { color: #50c878; }
    .result-label { font-size: 0.75rem; color: #8a8894; letter-spacing: 0.1em; text-transform: uppercase; }
    .risk-pill {
        display: inline-block;
        background: rgba(224,92,92,0.12);
        border: 1px solid rgba(224,92,92,0.25);
        color: #e05c5c; font-size: 0.78rem; font-weight: 500;
        padding: 0.22rem 0.7rem; border-radius: 100px; margin: 0.2rem;
    }
    .info-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #e05c5c;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1rem;
        font-size: 0.82rem; color: #8a8894;
        line-height: 1.6; margin-top: 1.2rem;
    }
    hr { border-color: #2a2a35 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL, SCALER, & FEATURES
# =====================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("selected_features.pkl")
        return model, scaler, features, True
    except Exception:
        return None, None, None, False

model, scaler, selected_features, model_loaded = load_artifacts()

# =====================================
# HEADER
# =====================================
st.markdown('<div class="hero-badge">🫀 BRFSS 2015 · CDC Dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Heart Disease<br><span class="hero-accent">Risk Assessment</span></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Answer 9 questions based on the CDC\'s Behavioral Risk Factor Surveillance System to estimate your cardiovascular risk.</p>', unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ **Model files not found.** Place `best_model.pkl`, `scaler.pkl`, and `selected_features.pkl` in the same directory.", icon="⚠️")

st.markdown("---")

# =====================================
# INPUTS
# =====================================
st.markdown('<div class="section-label">Clinical</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    HighBP = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with c2:
    HighChol = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

BMI = st.number_input("BMI", min_value=10.0, max_value=98.0, value=25.0, step=0.5, help="Body Mass Index. Healthy range: 18.5–24.9")

st.markdown('<div class="section-label">Health History</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    Smoker = st.selectbox("Smoked ≥100 cigarettes (lifetime)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with c4:
    Diabetes = st.selectbox("Diabetes", [0, 1, 2], format_func=lambda x: {0: "No", 1: "Pre-diabetic", 2: "Diabetic"}[x])

st.markdown('<div class="section-label">Lifestyle & Wellbeing</div>', unsafe_allow_html=True)

PhysActivity = st.selectbox("Physically active in past 30 days?", [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    help="Any exercise or physical activity outside of your job")

GenHlth = st.select_slider(
    "General Health",
    options=[1, 2, 3, 4, 5],
    value=3,
    format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x]
)

st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
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
with c6:
    Income = st.select_slider(
        "Household Income",
        options=list(range(1, 8)), # Adjusted upper bound to fit data mappings
        value=6,
        format_func=lambda x: {
            1: "< $10K", 2: "$10–15K", 3: "$15–20K", 4: "$20–25K",
            5: "$25–35K", 6: "$35–50K", 7: "≥ $50K"
        }[x]
    )

# =====================================
# PREDICT
# =====================================
predict_btn = st.button("🫀 Assess My Heart Disease Risk")

if predict_btn:
    # 1. Map Raw BMI to a Categorical Integer Range for BMI_Category
    if BMI < 18.5:
        BMI_Category = 1  # Underweight
    elif 18.5 <= BMI < 25.0:
        BMI_Category = 2  # Normal weight
    elif 25.0 <= BMI < 30.0:
        BMI_Category = 3  # Overweight
    else:
        BMI_Category = 4  # Obese

    # 2. Recreate the Health_Risk_Score calculation from training phase
    # (Aggregates standard categorical baseline risks)
    Health_Risk_Score = HighBP + HighChol + Smoker + (1 if Diabetes >= 1 else 0) + (1 if PhysActivity == 0 else 0)

    # 3. Restructure array exactly as model arrays demand
    # Sequence order specified in selected_features.pkl:
    # ['Health_Risk_Score', 'GenHlth', 'Age', 'HighBP', 'Diabetes', 'HighChol', 'Income', 'Smoker', 'PhysActivity', 'BMI_Category']
    input_data = np.array([[
        Health_Risk_Score, GenHlth, Age, HighBP, Diabetes, 
        HighChol, Income, Smoker, PhysActivity, BMI_Category
    ]], dtype=float)

    # UI tracking list
    risk_factors = []
    if HighBP == 1:       risk_factors.append("High Blood Pressure")
    if HighChol == 1:     risk_factors.append("High Cholesterol")
    if BMI >= 30:         risk_factors.append(f"BMI {BMI:.0f} (Obese)")
    if Smoker == 1:       risk_factors.append("Smoker")
    if Diabetes >= 1:     risk_factors.append("Diabetes / Pre-diabetic")
    if PhysActivity == 0: risk_factors.append("No Physical Activity")
    if GenHlth >= 4:      risk_factors.append("Poor General Health")
    if Age >= 9:          risk_factors.append("Age 60+")

    st.markdown("---")
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

    if model_loaded:
        try:
            input_scaled = scaler.transform(input_data)
            prediction  = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-title">⚠️ Elevated Risk Detected</div>
                    <div class="result-prob">{probability:.0%}</div>
                    <div class="result-label">Predicted risk probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-title">✅ Low Risk Profile</div>
                    <div class="result-prob">{probability:.0%}</div>
                    <div class="result-label">Predicted risk probability</div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        # Fallback rule-based summary
        risk_score = len(risk_factors)
        css_class = "result-high" if risk_score >= 3 else "result-low"
        emoji     = "⚠️" if risk_score >= 3 else "✅"
        level     = "Elevated Risk (Rule-Based)" if risk_score >= 3 else "Low / Moderate Risk"
        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-title">{emoji} {level}</div>
            <div class="result-prob">{risk_score}</div>
            <div class="result-label">Active risk factors (no model loaded)</div>
        </div>""", unsafe_allow_html=True)

    # Risk factor pills
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_factors:
        st.markdown("**Active Risk Factors**")
        pills = "".join(f'<span class="risk-pill">⚡ {rf}</span>' for rf in risk_factors)
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#50c878">✅ No major risk factors identified</span>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        This tool is <em>not</em> a medical diagnosis. Results are based on a model trained on
        CDC BRFSS 2015 survey data. Please consult a qualified healthcare professional for
        clinical evaluation.
    </div>""", unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#3a3a48; font-size:0.75rem; letter-spacing:0.05em">
    CDC BRFSS 2015 · For Educational & Research Use Only · Not a Medical Device
</p>""", unsafe_allow_html=True)