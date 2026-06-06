# 🩺 MediPredict AI — Multiple Disease Prediction System

<div align="center">

![MediPredict AI](https://i.postimg.cc/p5JxscjH/homepage-hero-section.png)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-brightgreen?style=for-the-badge)](https://multiple-disease-prediction-zv49.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

**An AI-powered web application that predicts the risk of Diabetes, Heart Disease, and Chronic Kidney Disease using trained Machine Learning models.**

[🔍 Live Demo](https://multiple-disease-prediction-zv49.onrender.com) • [📊 Features](#-features) • [🚀 Quick Start](#-quick-start) • [📁 Structure](#-project-structure)

</div>

---

## 📌 Project Overview

**MediPredict AI** is a machine learning-based multiple disease prediction system developed as a final year project. It uses clinical lab values as input and predicts disease risk using supervised ML models trained on real-world medical datasets.

The system predicts three major chronic diseases:
- 🩸 **Diabetes** — using the PIMA Indians Diabetes Dataset
- ❤️ **Heart Disease** — using the Cleveland Heart Disease Dataset  
- 🫘 **Kidney Disease (CKD)** — using the UCI Chronic Kidney Disease Dataset

---

## 🌐 Live Demo

> **Deployed on Render:** [https://multiple-disease-prediction-zv49.onrender.com](https://multiple-disease-prediction-zv49.onrender.com)

---

## 🖼️ Screenshots

### 🏠 Homepage — Hero Section
![Homepage](https://i.postimg.cc/p5JxscjH/homepage-hero-section.png)

### 📖 How It Works Section
![How It Works](https://i.postimg.cc/4n6J4ySc/how-it-works-section.png)

### 🩸 Diabetes Prediction — Input Form
![Diabetes Form](https://i.postimg.cc/yJ2VxZBq/diabetes-prediction-form.png)

### 🩸 Diabetes Prediction — Result
![Diabetes Result](https://i.postimg.cc/hQ1g2ZdZ/diabetes-prediction-result.png)

### ❤️ Heart Disease Prediction Page
![Heart Prediction](https://i.postimg.cc/bS98mVtK/heart-prediction-page.png)

### 🫘 Kidney Disease Prediction Page
![Kidney Prediction](https://i.postimg.cc/F7hhYmt4/kidney-prediction-page.png)

---

## ✨ Features

- 🔬 **3 Disease Predictions** — Diabetes, Heart Disease, Chronic Kidney Disease
- 🤖 **6+ ML Models Compared** — Best performing model selected per disease
- 📊 **Visual Risk Charts** — Horizontal bar chart showing risk factor breakdown
- 🚨 **Alert System** — Color-coded alerts (High / Moderate / Low risk)
- 💡 **Smart Hints** — Every input field has normal range, example, and explanation
- 📱 **Responsive Design** — Works on desktop, tablet, and mobile
- 🤖 **MediBot Chatbot** — AI assistant to explain any medical field
- 🎬 **Animated UI** — YouTube video background, floating particles, smooth transitions
- ⚡ **Demo Mode** — Works without Flask using calibrated clinical scoring
- 🔒 **Privacy First** — No data stored, all processing done locally

---

## 🧠 Machine Learning Models

| Disease | Dataset | Best Model | Accuracy | Features |
|---------|---------|-----------|----------|----------|
| 🩸 Diabetes | PIMA Indians (768 samples) | SVM (SVC) | **82%** | 8 |
| ❤️ Heart Disease | Cleveland (303 samples) | Random Forest | **90%** | 13 |
| 🫘 Kidney Disease | UCI CKD (400 samples) | Random Forest | **98%** | 24 |

### Models Compared Per Disease
`SVM` · `Random Forest` · `Logistic Regression` · `KNN` · `Decision Tree` · `Naive Bayes` · `XGBoost`

---

## 📁 Project Structure

```
Multiple_Disease_Prediction/
│
├── 📂 Datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── kidney_disease.csv
│
├── 📂 Notebooks/
│   ├── diabetes_Disease_training.ipynb
│   ├── Heart_Disease_Training.ipynb
│   └── Kidney_Disease_training.ipynb
│
├── 📂 models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── kidney_model.pkl
│
├── 📂 static/
│   ├── Diesease_Background.mp4
│   └── medical_photo.png
│
├── 📂 templates/
│   ├── index.html          ← Homepage
│   ├── diabetes.html       ← Diabetes prediction page
│   ├── heart.html          ← Heart disease prediction page
│   └── kidney.html         ← Kidney disease prediction page
│
├── app.py                  ← Flask backend
├── requirements.txt
├── Procfile                ← Render/Heroku deployment
└── runtime.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/sUmiT-BhAtT-443/Multiple_Disease_Prediction.git
cd Multiple_Disease_Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models (if `.pkl` files not present)
Open and run all cells in each notebook:
```
Notebooks/diabetes_Disease_training.ipynb
Notebooks/Heart_Disease_Training.ipynb
Notebooks/Kidney_Disease_training.ipynb
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

---

## 📦 Requirements

```txt
flask
flask-cors
scikit-learn
numpy
pandas
gunicorn
```

Install all:
```bash
pip install flask flask-cors scikit-learn numpy pandas gunicorn
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage |
| `GET` | `/diabetes` | Diabetes prediction page |
| `GET` | `/heart` | Heart disease prediction page |
| `GET` | `/kidney` | Kidney disease prediction page |
| `POST` | `/predict/diabetes` | Diabetes prediction API |
| `POST` | `/predict/heart` | Heart disease prediction API |
| `POST` | `/predict/kidney` | Kidney disease prediction API |
| `GET` | `/health` | Server health check |

### Example API Request
```bash
curl -X POST http://localhost:5000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{"pregnancies":2,"glucose":120,"bloodpressure":70,"skinthickness":20,"insulin":80,"bmi":25,"dpf":0.5,"age":35}'
```

### Example API Response
```json
{
  "prediction": 0,
  "result": "Not Diabetic",
  "risk_percent": 22.0
}
```

---

## 🧪 Model Preprocessing Details

### Diabetes (SVM)
- Preprocessing: `StandardScaler` → `SVC(probability=True)`
- Saved as `sklearn.pipeline.Pipeline` for end-to-end scaling

### Heart Disease (Random Forest)
- Preprocessing: Log transform on `trestbps`, `chol`, `thalach`
- `RandomForestClassifier` — no scaler needed

### Kidney Disease (Random Forest)  
- Preprocessing: `LabelEncoder` (alphabetical order)
- Encodings: `rbc/pc` → normal=1, abnormal=0 | `htn/dm/cad/pe/ane` → yes=1, no=0
- Class label: `ckd=0`, `notckd=1`

---

## 🌐 Deployment on Render

This project is deployed on **Render** using:

**`Procfile`:**
```
web: gunicorn app:app
```

**`runtime.txt`:**
```
python-3.10.0
```

**Environment:** Web Service → Free Plan → Auto Deploy from GitHub

---

## 🎯 Aim & Objectives

### Aim
To develop an intelligent web-based system that predicts the risk of multiple diseases — Diabetes, Heart Disease, and Chronic Kidney Disease — using supervised machine learning algorithms trained on clinical datasets.

### Objectives
1. Collect and preprocess real-world clinical datasets for three diseases
2. Train and compare 6+ ML algorithms and select the best-performing model per disease
3. Build a user-friendly web interface with clear field explanations and normal ranges
4. Deploy a Flask REST API backend integrating trained models with the frontend
5. Provide personalized health precautions and risk factor visual breakdowns

### Problem Statement
Chronic diseases often go undetected until an advanced stage due to lack of regular screening, high costs, and limited healthcare access. This project addresses the need for an automated, accessible disease prediction system using standard clinical parameters to enable early risk identification.

---

## ⚠️ Disclaimer

> This application is developed for **educational and academic purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Developer

**Sumit Bhatt**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sumit-bhatt-352ba9334)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/sUmiT-BhAtT-443)

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

```
git clone https://github.com/sUmiT-BhAtT-443/Multiple_Disease_Prediction.git
```

---

<div align="center">
Made with ❤️ by <a href="https://github.com/sUmiT-BhAtT-443">Sumit Bhatt</a>
</div>
