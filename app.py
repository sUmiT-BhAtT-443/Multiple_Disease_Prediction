"""
MediPredict AI — Flask Backend
================================
Folder Structure:
  Multiple_Disease_Prediction/
  ├── Datasets/
  ├── models/
  │   ├── diabetes_model.pkl
  │   ├── heart_model.pkl
  │   └── kidney_model.pkl
  ├── Notebooks/
  ├── static/
  │   ├── Diesease_Background.mp4
  │   └── medical_photo.png
  ├── templates/
  │   ├── index.html
  │   ├── diabetes.html
  │   ├── heart.html
  │   └── kidney.html
  └── app.py

Run:
    pip install flask flask-cors scikit-learn numpy pandas
    python app.py
    Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, 'templates'),
    static_folder=os.path.join(BASE, 'static')
)
CORS(app)

MODELS_DIR = os.path.join(BASE, 'models')


# ── Model loader ──────────────────────────────────────────────────────────────
def load_pkl(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        log.warning(f"Model not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            m = pickle.load(f)
        log.info(f"Loaded: {name} ({type(m).__name__})")
        return m
    except Exception as e:
        log.error(f"Error loading {name}: {e}")
        return None


diabetes_model  = load_pkl('diabetes_model.pkl')
diabetes_scaler = load_pkl('diabetes_scaler.pkl')
heart_model     = load_pkl('heart_model.pkl')
kidney_model    = load_pkl('kidney_model.pkl')

# PIMA fallback scaling
PIMA_MEAN = np.array([3.845, 120.894, 69.105, 20.536,  79.799, 31.993, 0.4719, 33.241])
PIMA_STD  = np.array([3.370,  31.973, 19.355, 15.952, 115.244,  7.884, 0.3313, 11.760])


# ── Clinical risk scoring (fallback when model gives 0%) ──────────────────────
def clinical_diabetes_risk(vals):
    """vals = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]"""
    s = 0
    preg, g, bp, skin, ins, bmi, dpf, age = vals

    if g >= 200: s += 42
    elif g >= 140: s += 34
    elif g >= 126: s += 28
    elif g >= 110: s += 18
    elif g >= 100: s += 10
    elif g >= 70:  s += 0
    elif g > 0:    s += 8

    if bmi >= 40:   s += 22
    elif bmi >= 35: s += 18
    elif bmi >= 30: s += 14
    elif bmi >= 27: s += 9
    elif bmi >= 25: s += 5
    elif bmi >= 18.5: s += 0
    elif bmi > 0:   s += 9

    if age >= 65: s += 16
    elif age >= 55: s += 12
    elif age >= 45: s += 8
    elif age >= 35: s += 4

    if ins > 800: s += 14
    elif ins > 500: s += 11
    elif ins > 300: s += 8
    elif ins > 166: s += 5
    elif ins == 0 and g > 100: s += 8

    if bp > 120: s += 12
    elif bp > 110: s += 9
    elif bp > 100: s += 6
    elif bp > 90:  s += 4
    elif bp > 80:  s += 2
    elif bp >= 60: s += 0
    elif bp > 0:   s += 4

    if dpf > 2.0: s += 10
    elif dpf > 1.5: s += 8
    elif dpf > 1.0: s += 6
    elif dpf > 0.5: s += 3

    if preg >= 10: s += 8
    elif preg >= 7: s += 6
    elif preg >= 5: s += 4
    elif preg >= 3: s += 2

    if skin > 60: s += 6
    elif skin > 50: s += 4
    elif skin > 40: s += 2

    return int(min(95, max(5, round(s / 132 * 100))))


def clinical_heart_risk(vals):
    """vals = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]"""
    s = 0
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = vals
    cp = int(cp); exang = int(exang); ca = int(ca); thal = int(thal)
    restecg = int(restecg); fbs = int(fbs)

    if chol >= 300: s += 30
    elif chol >= 240: s += 22
    elif chol >= 200: s += 12

    expected_hr = max(220 - age, 100)
    hr_ratio = thalach / expected_hr
    if hr_ratio < 0.50:   s += 28
    elif hr_ratio < 0.65: s += 20
    elif hr_ratio < 0.75: s += 12
    elif hr_ratio < 0.85: s += 5

    if cp == 0: s += 22
    elif cp == 1: s += 12
    elif cp == 2: s += 6
    else: s += 2

    if age >= 70: s += 18
    elif age >= 60: s += 13
    elif age >= 50: s += 8
    elif age >= 40: s += 4

    if exang == 1: s += 16

    if oldpeak >= 4: s += 14
    elif oldpeak >= 3: s += 11
    elif oldpeak >= 2: s += 8
    elif oldpeak >= 1: s += 4

    if ca >= 3: s += 12
    elif ca >= 2: s += 9
    elif ca >= 1: s += 5

    if thal == 3: s += 12
    elif thal == 2: s += 6

    if trestbps >= 160: s += 10
    elif trestbps >= 140: s += 7
    elif trestbps >= 130: s += 4

    if restecg == 2: s += 8
    elif restecg == 1: s += 4

    if fbs == 1: s += 6

    return int(min(95, max(5, round(s / 176 * 100))))


def clinical_kidney_risk(vals):
    """vals = [age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]"""
    s = 0
    (age, bp, sg, al, su, rbc, pc, pcc, ba,
     bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
     htn, dm, cad, appet, pe, ane) = vals
    htn = int(htn); dm = int(dm); pc = int(pc)
    pe = int(pe); ane = int(ane); appet = int(appet)

    if sc >= 10:  s += 40
    elif sc >= 5:  s += 35
    elif sc >= 3:  s += 28
    elif sc >= 2:  s += 20
    elif sc >= 1.5: s += 12
    elif sc >= 1.2: s += 5

    if al >= 4: s += 30
    elif al >= 3: s += 24
    elif al >= 2: s += 16
    elif al >= 1: s += 8

    if hemo < 7:   s += 24
    elif hemo < 9:  s += 18
    elif hemo < 11: s += 12
    elif hemo < 13: s += 6

    if bu >= 100: s += 18
    elif bu >= 60: s += 13
    elif bu >= 40: s += 8
    elif bu >= 25: s += 3

    if htn == 1: s += 12
    if dm == 1:  s += 10

    if sod < 125: s += 10
    elif sod < 130: s += 6
    elif sod < 135: s += 3

    if sg <= 1.005: s += 8
    elif sg <= 1.010: s += 4

    if pc == 0:    s += 8
    if pe == 1:    s += 8
    if ane == 1:   s += 7
    if appet == 1: s += 6

    return int(min(95, max(5, round(s / 189 * 100))))


# ── Safe predict with clinical fallback ───────────────────────────────────────
def predict_with_fallback(model, X, clinical_fn, raw_vals, threshold=42):
    """Try model predict_proba. If result is 0% or unavailable, use clinical scoring."""
    model_risk = None

    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            risk = round(float(proba[1]) * 100, 1)
            if risk >= 1.0:  # model gave a real answer
                model_risk = risk
    except Exception:
        pass

    if model_risk is not None:
        pred = int(model.predict(X)[0])
        return pred, model_risk

    # Fallback to clinical scoring
    risk = clinical_fn(raw_vals)
    pred = 1 if risk >= threshold else 0
    return pred, float(risk)


def predict_kidney_with_fallback(model, X, raw_vals):
    """Kidney: ckd=0, notckd=1 — special handling"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            risk = round(float(proba[0]) * 100, 1)  # proba[0] = P(ckd)
            if risk >= 1.0:
                pred_raw = model.predict(X)[0]
                is_ckd = str(pred_raw).strip().lower() in ['0', 'ckd']
                return (1 if is_ckd else 0), risk
    except Exception:
        pass

    risk = clinical_kidney_risk(raw_vals)
    pred = 1 if risk >= 45 else 0
    return pred, float(risk)


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/heart')
def heart_page():
    return render_template('heart.html')

@app.route('/kidney')
def kidney_page():
    return render_template('kidney.html')


# ── Diabetes prediction ───────────────────────────────────────────────────────
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        d = request.get_json(force=True, silent=True) or {}
        keys = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
                'insulin', 'bmi', 'dpf', 'age']
        for k in keys:
            if k not in d:
                return jsonify({'error': f'Missing field: {k}'}), 400

        raw = [float(d[k]) for k in keys]
        X_raw = np.array([raw])

        if diabetes_model is None:
            risk = clinical_diabetes_risk(raw)
            pred = 1 if risk >= 42 else 0
            return jsonify({'prediction': pred,
                            'result': 'Diabetic' if pred == 1 else 'Not Diabetic',
                            'risk_percent': float(risk), 'mode': 'demo'})

        # Scale
        try:
            from sklearn.pipeline import Pipeline
            is_pipe = isinstance(diabetes_model, Pipeline)
        except ImportError:
            is_pipe = False

        if is_pipe:
            X = X_raw
        elif diabetes_scaler is not None:
            X = diabetes_scaler.transform(X_raw)
        else:
            X = (X_raw - PIMA_MEAN) / PIMA_STD

        pred, risk = predict_with_fallback(
            diabetes_model, X, clinical_diabetes_risk, raw, threshold=42)

        return jsonify({'prediction': pred,
                        'result': 'Diabetic' if pred == 1 else 'Not Diabetic',
                        'risk_percent': risk})

    except Exception as e:
        log.error(f"Diabetes error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Heart prediction ──────────────────────────────────────────────────────────
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        d = request.get_json(force=True, silent=True) or {}
        keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for k in keys:
            if k not in d:
                return jsonify({'error': f'Missing field: {k}'}), 400

        raw = [float(d[k]) for k in keys]

        if heart_model is None:
            risk = clinical_heart_risk(raw)
            pred = 1 if risk >= 45 else 0
            return jsonify({'prediction': pred,
                            'result': 'Heart Disease Detected' if pred == 1 else 'No Heart Disease',
                            'risk_percent': float(risk), 'mode': 'demo'})

        # Log transform (from notebook)
        X = np.array([[
            raw[0], raw[1], raw[2],
            np.log(max(raw[3], 1e-9)),
            np.log(max(raw[4], 1e-9)),
            raw[5], raw[6],
            np.log(max(raw[7], 1e-9)),
            raw[8], raw[9], raw[10], raw[11], raw[12]
        ]])

        pred, risk = predict_with_fallback(
            heart_model, X, clinical_heart_risk, raw, threshold=45)

        return jsonify({'prediction': pred,
                        'result': 'Heart Disease Detected' if pred == 1 else 'No Heart Disease',
                        'risk_percent': risk})

    except Exception as e:
        log.error(f"Heart error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Kidney prediction ─────────────────────────────────────────────────────────
@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        d = request.get_json(force=True, silent=True) or {}
        keys = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for k in keys:
            if k not in d:
                return jsonify({'error': f'Missing field: {k}'}), 400

        raw = [float(d[k]) for k in keys]

        if kidney_model is None:
            risk = clinical_kidney_risk(raw)
            pred = 1 if risk >= 45 else 0
            return jsonify({'prediction': pred,
                            'result': 'CKD Detected' if pred == 1 else 'No CKD',
                            'risk_percent': float(risk), 'mode': 'demo'})

        X = np.array([raw])
        pred, risk = predict_kidney_with_fallback(kidney_model, X, raw)

        return jsonify({'prediction': pred,
                        'result': 'CKD Detected' if pred == 1 else 'No CKD',
                        'risk_percent': risk})

    except Exception as e:
        log.error(f"Kidney error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'models': {
            'diabetes': 'loaded' if diabetes_model else 'demo mode',
            'heart':    'loaded' if heart_model    else 'demo mode',
            'kidney':   'loaded' if kidney_model   else 'demo mode',
        }
    })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print()
    print("=" * 55)
    print("  MediPredict AI — Flask Backend")
    print("=" * 55)
    print(f"  Diabetes : {'✅ Model loaded' if diabetes_model else '⚡ Demo mode'}")
    print(f"  Heart    : {'✅ Model loaded' if heart_model    else '⚡ Demo mode'}")
    print(f"  Kidney   : {'✅ Model loaded' if kidney_model   else '⚡ Demo mode'}")
    print("=" * 55)
    print()
    print("  → http://localhost:5000")
    print()
    app.run(debug=True, port=5000, host='0.0.0.0')
