🧠 Alzheimer's Predictor & Diagnosis Platform
🌟 Project Overview
The Alzheimer's Predictor Platform is an interactive, Arabic-first medical screening and awareness web application built using Streamlit and Machine Learning (Random Forest).

Designed and developed by Raneen Ashraf (رنين أشرف), the platform serves as an end-to-end clinical screening assistant that evaluates a patient's cognitive and lifestyle risk factors, classifies the likelihood of Alzheimer's Disease, generates downloadable medical PDF reports, and connects patients with nearby specialist neurologists using interactive geolocation maps.

🏗️ Technical Architecture & Workflow
┌─────────────────────────────────────────┐
 1. Clinical & Lifestyle Input │  10 Top Selected Features (MMSE, ADL,   │
    (Interactive Form)       ──▶  Sleep, Diet, BP, HDL, Functional Assessment)
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
 2. ML Classification Engine   │  Random Forest Classifier + SelectKBest │
    (Scikit-Learn)           ──▶  Predicts Diagnosis & Probability Score │
                               └────────────────────┬────────────────────┘
                                                    │
                               ┌────────────────────┴────────────────────┐
                               ▼                                         ▼
           ┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
 3. Output │   Automated Arabic PDF Medical Report │ │ Interactive Risk Heatmap & Diagnostic │
           │   (ReportLab with Risk Recommendations) │ │ Visualizations (Seaborn / Matplotlib) │
           └───────────────────────────────────────┘ └───────────────────────────────────────┘
✨ Core Features & Application Tabs
The platform is organized into 5 interactive tabs:

1. 📝 السجل الطبي للمريض (Patient Information Tab)
Captures essential demographic data: Full Name, Age, Gender, and Contact Number.
Features strict validation for Egyptian phone numbers (010, 011, 012, 015 with 11-digit verification).
Stores session information persistently across tabs using st.session_state.
2. 🔮 التشخيص وتقييم الخطورة (Diagnosis & Risk Assessment Tab)
Collects inputs for the Top 10 Clinical & Lifestyle Predictors:
Cognitive & Behavioral: MMSE Score (Mini-Mental State Exam), Functional Assessment, Memory Complaints, Behavioral Problems, ADL (Activities of Daily Living).
Biomarkers & Lifestyle: Hypertension (ضغط الدم), Cholesterol HDL, BMI, Sleep Quality, Diet Quality.
Real-Time Risk Probability: Predicts either "سليم" (Healthy) or "تم اكتشاف الزهايمر" (Positive Detection) along with exact risk probabilities categorized into:
🟢 Low Risk (< 40%)
🔠 Moderate Risk (40% - 69%)
🔴 High Risk (≥ 70%)
📄 Downloadable Medical PDF Report: Uses ReportLab to generate a beautifully formatted Arabic PDF report (generate_arabic_pdf) featuring the patient's demographics, diagnosis, detected risk factors, and actionable medical recommendations.
3. 📊 تقييم أداء النموذج (Model Evaluation Dashboard Tab)
Displays transparent validation metrics from the trained machine learning pipeline:
Sensitivity (الحساسية): ~82.3%
Specificity (الخصوصية): ~89.1%
Renders an interactive Confusion Matrix Heatmap and full multi-class classification report so medical reviewers can understand the model's reliability.
4. 🌍 الأطباء المتخصصون (Find Nearby Specialists & Interactive Map Tab)
Integrates Geopy (Nominatim) and Folium (streamlit-folium) to calculate geodesic distances (km) between the patient's selected Governorate/Area (e.g., Cairo, Alexandria) and specialized neurologists/memory clinics.
Displays interactive map markers with tooltips showing doctor specialties, clinic addresses, contact numbers, and working hours sorted by closest distance.
5. 🧠 موسوعة معلومات الزهايمر (Alzheimer's Medical Encyclopedia Tab)
Offers structured educational sub-tabs covering:
تعريف المرض (Definition): Progression timeline and statistics.
الأعراض (Symptoms): Early-stage vs. advanced signs.
الوقاية (Prevention): Actionable lifestyle modifications (Mediterranean diet, 30-min daily exercise, cognitive stimulation).
العلاج (Treatments): Overview of available pharmacological treatments (Cholinesterase inhibitors, Memantine) and supportive therapies.
🛠️ Technology Stack
Layer	Technologies Used
Web Interface	Streamlit (st.tabs, st.session_state, custom CSS layouts)
Machine Learning Core	Scikit-Learn (RandomForestClassifier, SelectKBest, StandardScaler, LabelEncoder)
Data Processing	Pandas, NumPy, Joblib
Data Visualization	Seaborn, Matplotlib, Plotly
Mapping & Geolocation	Folium, Streamlit-Folium, Geopy (geodesic, Nominatim)
Report Generation	ReportLab (Custom Arabic TTF rendering & dynamic canvas drawing)
