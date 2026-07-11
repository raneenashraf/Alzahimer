# 🧠 Alzheimer's Predictor & Diagnosis Platform

An AI-powered, Arabic-first medical screening platform that predicts Alzheimer's disease risk using Machine Learning and provides an interactive clinical decision-support system.

Developed by **Raneen Ashraf** using **Python, Streamlit, and Scikit-Learn**.

---

## 🌟 Project Overview

The **Alzheimer's Predictor & Diagnosis Platform** is an end-to-end web application designed to assist in the early screening of Alzheimer's disease. The system analyzes cognitive, clinical, and lifestyle factors to estimate a patient's risk level using a trained **Random Forest Classifier**.

Beyond prediction, the platform generates downloadable Arabic medical reports, visualizes model performance, and helps patients locate nearby neurologists through interactive maps.

---

## 🚀 Features

### 📝 Patient Information
- Collect patient demographic information.
- Validate Egyptian phone numbers.
- Store session data across the application using `st.session_state`.

### 🔮 Alzheimer's Prediction
Evaluate patients using the top clinical and lifestyle indicators including:

- MMSE Score
- Functional Assessment
- Activities of Daily Living (ADL)
- Memory Complaints
- Behavioral Problems
- Hypertension
- HDL Cholesterol
- BMI
- Sleep Quality
- Diet Quality

The model predicts:

- 🟢 Low Risk
- 🟡 Moderate Risk
- 🔴 High Risk

with prediction probabilities.

---

### 📄 Arabic Medical PDF Report

Generate a professional Arabic medical report containing:

- Patient Information
- Prediction Result
- Risk Probability
- Identified Risk Factors
- Medical Recommendations

Built using **ReportLab** with Arabic font rendering.

---

### 📊 Model Evaluation Dashboard

Visualize model performance through:

- Accuracy Metrics
- Sensitivity
- Specificity
- Confusion Matrix
- Classification Report

using **Matplotlib** and **Seaborn**.

---

### 🌍 Find Nearby Neurologists

Locate specialized doctors based on the patient's location using:

- Folium Maps
- Geopy
- Nominatim Geocoding

The application displays:

- Doctor Name
- Clinic Address
- Phone Number
- Working Hours
- Distance from Patient

---

### 📚 Alzheimer's Medical Encyclopedia

Provides educational resources covering:

- Disease Definition
- Symptoms
- Risk Factors
- Prevention
- Available Treatments
- Lifestyle Recommendations

---

# 🏗️ System Architecture

```
Patient Information
        │
        ▼
Clinical & Lifestyle Assessment
        │
        ▼
Feature Selection (Top 10 Features)
        │
        ▼
Random Forest Classifier
        │
        ▼
Prediction & Risk Probability
        │
 ┌──────┴──────────────┐
 ▼                     ▼
PDF Report      Interactive Dashboard
        │
        ▼
Nearby Neurologists Map
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Model | Random Forest Classifier |
| Feature Selection | SelectKBest |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Mapping | Folium, Streamlit-Folium |
| Geolocation | Geopy, Nominatim |
| PDF Generation | ReportLab |
| Model Persistence | Joblib |

---

# 🤖 Machine Learning Pipeline

- Data Preprocessing
- Feature Selection using SelectKBest
- Feature Scaling
- Random Forest Training
- Probability Prediction
- Risk Classification
- Report Generation

---

# 📈 Model Performance

The trained model provides:

- **Sensitivity:** ~82.3%
- **Specificity:** ~89.1%

Evaluation includes:

- Confusion Matrix
- Classification Report
- Prediction Probabilities

---

# 📂 Project Structure

```
Alzheimers-Predictor/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── data/
├── assets/
├── reports/
├── doctors_data/
├── requirements.txt
└── README.md
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Alzheimers-Predictor.git
```

Navigate to the project

```bash
cd Alzheimers-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📷 Application Modules

- 📝 Patient Information
- 🔮 Risk Assessment
- 📄 Medical PDF Report
- 📊 Model Evaluation Dashboard
- 🌍 Nearby Specialists Map
- 📚 Alzheimer's Medical Encyclopedia

---

# 🎯 Key Highlights

- Arabic-first medical interface
- Machine Learning-powered diagnosis
- Random Forest classification
- Interactive dashboards
- Downloadable Arabic PDF reports
- Doctor recommendation system
- Geolocation-based clinic search
- Educational medical encyclopedia
- Responsive Streamlit interface

---

# 👩‍💻 Developer

**Raneen Ashraf**

📧 Email: raneenashraf477@gmail.com

💼 LinkedIn: https://linkedin.com/in/raneenashraf/

💻 GitHub: https://github.com/raneenashraf

---

## ⭐ If you found this project useful, don't forget to star the repository!
