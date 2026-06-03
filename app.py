"""
app.py – Heart Disease Prediction Web Application
Deployment step of the Final Data Science Project
Run: python app.py  →  http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# ── Load saved artifacts
model    = joblib.load('best_model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('selected_features.pkl')

# ── HTML template (single-file app, no separate templates folder needed)
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heart Disease Risk Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; min-height: 100vh;
           display: flex; align-items: center; justify-content: center; padding: 20px; }
    .card { background: white; border-radius: 16px; padding: 36px 40px;
            max-width: 600px; width: 100%; box-shadow: 0 8px 32px rgba(0,0,0,0.12); }
    h1 { color: #c0392b; font-size: 1.8rem; margin-bottom: 6px; }
    .subtitle { color: #666; margin-bottom: 28px; font-size: 0.95rem; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .form-group { display: flex; flex-direction: column; gap: 5px; }
    label { font-size: 0.85rem; font-weight: 600; color: #444; }
    select, input { padding: 9px 12px; border: 1.5px solid #ddd; border-radius: 8px;
                    font-size: 0.95rem; color: #333; transition: border-color 0.2s; }
    select:focus, input:focus { outline: none; border-color: #e74c3c; }
    .btn { margin-top: 24px; width: 100%; padding: 13px; background: #c0392b;
           color: white; border: none; border-radius: 10px; font-size: 1.05rem;
           font-weight: 700; cursor: pointer; transition: background 0.2s; }
    .btn:hover { background: #a93226; }
    .result { margin-top: 20px; padding: 18px; border-radius: 10px; text-align: center;
              font-size: 1.1rem; font-weight: 600; display: none; }
    .result.high { background: #fde8e8; color: #c0392b; border: 2px solid #e74c3c; }
    .result.low  { background: #e8f8f0; color: #196f3d; border: 2px solid #27ae60; }
    .prob-bar-wrap { margin-top: 10px; background: #eee; border-radius: 20px; height: 12px; overflow: hidden; }
    .prob-bar { height: 100%; border-radius: 20px; transition: width 0.6s ease; }
    .features-note { font-size: 0.78rem; color: #888; margin-top: 20px; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p class="subtitle">Enter your health indicators to get an AI-powered risk assessment.</p>

    <div class="form-grid">
      <div class="form-group">
        <label>High Blood Pressure</label>
        <select name="HighBP">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>High Cholesterol</label>
        <select name="HighChol">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>BMI</label>
        <input type="number" name="BMI" value="25" min="10" max="80" step="0.1">
      </div>
      <div class="form-group">
        <label>Smoker (≥100 cigarettes lifetime)</label>
        <select name="Smoker">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>Diabetes</label>
        <select name="Diabetes">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>Physical Activity (last 30 days)</label>
        <select name="PhysActivity">
          <option value="1">Yes</option>
          <option value="0">No</option>
        </select>
      </div>
      <div class="form-group">
        <label>General Health (1=Excellent, 5=Poor)</label>
        <select name="GenHlth">
          <option value="1">1 – Excellent</option>
          <option value="2">2 – Very Good</option>
          <option value="3" selected>3 – Good</option>
          <option value="4">4 – Fair</option>
          <option value="5">5 – Poor</option>
        </select>
      </div>
      <div class="form-group">
        <label>Age Group (1=18-24 … 13=80+)</label>
        <select name="Age">
          {% for i in range(1,14) %}
          <option value="{{ i }}" {% if i==7 %}selected{% endif %}>{{ i }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="form-group">
        <label>Income Group (1=<$10K … 8=>$75K)</label>
        <select name="Income">
          {% for i in range(1,9) %}
          <option value="{{ i }}" {% if i==5 %}selected{% endif %}>{{ i }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button class="btn" onclick="predict()">🔍 Predict Risk</button>

    <div class="result" id="result">
      <div id="result-text"></div>
      <div class="prob-bar-wrap" style="margin-top:10px">
        <div class="prob-bar" id="prob-bar"></div>
      </div>
      <div id="prob-label" style="font-size:0.85rem; margin-top:6px; color:#555"></div>
    </div>
    <p class="features-note">This tool uses a Logistic Regression model trained on the BRFSS 2015 dataset (253,680 records).<br>For educational purposes only — not a medical diagnosis.</p>
  </div>

  <script>
    async function predict() {
      const fields = ['HighBP','HighChol','BMI','Smoker','Diabetes',
                      'PhysActivity','GenHlth','Age','Income'];
      const data = {};
      fields.forEach(f => {
        const el = document.querySelector(`[name="${f}"]`);
        data[f] = parseFloat(el.value);
      });

      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(data)
      });
      const res = await resp.json();

      const div = document.getElementById('result');
      const txt = document.getElementById('result-text');
      const bar = document.getElementById('prob-bar');
      const lbl = document.getElementById('prob-label');

      div.style.display = 'block';
      const pct = (res.probability * 100).toFixed(1);
      if (res.prediction === 1) {
        div.className = 'result high';
        txt.textContent = '⚠️ Higher Risk of Heart Disease Detected';
        bar.style.background = '#e74c3c';
      } else {
        div.className = 'result low';
        txt.textContent = '✅ Lower Risk of Heart Disease';
        bar.style.background = '#27ae60';
      }
      bar.style.width = pct + '%';
      lbl.textContent = `Model confidence: ${pct}% probability of heart disease`;
    }
  </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Render the prediction form."""
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive JSON with 9 health indicators, compute engineered features,
    scale, and return prediction + probability.
    """
    data = request.get_json()

    # ── Replicate feature engineering from training
    bmi = float(data.get('BMI', 25))
    bmi_clipped = min(bmi, 80)

    # Health_Risk_Score
    health_risk = (
        int(data['HighBP']) +
        int(data['HighChol']) +
        int(data['Smoker']) +
        int(data['Diabetes']) +
        (1 - int(data['PhysActivity']))
    )

    # BMI_Category
    if bmi_clipped < 18.5:
        bmi_cat = 0
    elif bmi_clipped < 25:
        bmi_cat = 1
    elif bmi_clipped < 30:
        bmi_cat = 2
    else:
        bmi_cat = 3

    # Build full feature dict (matches training columns)
    row = {
        'HighBP':             int(data['HighBP']),
        'HighChol':           int(data['HighChol']),
        'BMI':                bmi_clipped,
        'Smoker':             int(data['Smoker']),
        'Diabetes':           int(data['Diabetes']),
        'PhysActivity':       int(data['PhysActivity']),
        'GenHlth':            float(data['GenHlth']),
        'Age':                float(data['Age']),
        'Income':             float(data['Income']),
        'Health_Risk_Score':  health_risk,
        'BMI_Category':       bmi_cat,
    }

    # Select features in the order used during training
    X = np.array([[row[f] for f in features]])
    X_sc = scaler.transform(X)

    prediction  = int(model.predict(X_sc)[0])
    probability = float(model.predict_proba(X_sc)[0][1])

    return jsonify({'prediction': prediction, 'probability': round(probability, 4)})


@app.route('/health')
def health():
    """Simple health-check endpoint."""
    return jsonify({'status': 'ok', 'model': 'Tuned Logistic Regression',
                    'features': features})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)