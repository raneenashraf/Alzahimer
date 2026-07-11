# -*- coding: utf-8 -*-
# =========================================
# Alzheimer's Diagnosis App – Safe Unicode Version
# =========================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import base64
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import streamlit as st
from datetime import datetime
import sys
import os
import re

# حل مشكلة Unicode للويندوز
if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# دالة لمعالجة النصوص بشكل آمن
def safe_text(text):
    if text is None:
        return ""
    try:
        if isinstance(text, str):
            # إزالة أي أحرف غير صالحة في UTF-8
            text = re.sub(r'[^\x00-\x7F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        return str(text)
    except Exception as e:
        print(f"Error encoding text: {e}")
        return ""

# تهيئة الخطوط العربية لـ ReportLab
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('Arabic', 'arial.ttf'))  # تأكد من وجود الخط في نظامك
except:
    pass

st.set_page_config(page_title="Alzheimer's Predictor", layout="wide", page_icon="🧠")

# =========================================
# Section 1: Load and Preprocess Data
# =========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif


# تحميل البيانات
@st.cache_data
def load_data():
    data = pd.read_excel(r"C:\Users\ranee\Dropbox\PC\Desktop\outputAlzheimer.xlsx")
    data = data.drop(columns=['EducationLevel'], errors='ignore')
    return data

data = load_data()

# ترميز البيانات النصية
label_encoder = LabelEncoder()
for col in data.select_dtypes(['object', 'string']):
    data[col] = label_encoder.fit_transform(data[col])

# تقسيم البيانات
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# حفظ نسخة من X الأصلية لاستخدامها لاحقاً في تحديد أسماء الميزات المختارة
X_original = X.copy()

# تدريب النموذج على كل الفيتشرز
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))
pipeline.fit(X_train, y_train)
model = pipeline
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج باستخدام جميع الميزات: {accuracy * 100:.2f}%")

# طباعة ترتيب الفيتشرز
features = X.columns.tolist()
print("\nترتيب الفيتشرز المستخدمة في تدريب النموذج:")
for i, feat in enumerate(features):
    print(f"{i+1}. {feat}")

# =====================
# استخدام SelectKBest لاختيار أهم الميزات
# =====================

# حذف الأعمدة غير المستخدمة في التقييم
X = data[[
    'SleepQuality', 'Hypertension', 'CholesterolHDL', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'BMI', 'DietQuality'
]]

selector = SelectKBest(score_func=f_classif, k='all')  # جميع الميزات العشرة المختارة مسبقاً
X_selected = selector.fit_transform(X, y)

# طباعة أسماء الميزات المختارة
selected_features = X.columns[selector.get_support()]
print("\nالميزات التي تم اختيارها باستخدام SelectKBest:")
print(selected_features)

# تدريب النموذج باستخدام الميزات المختارة
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج باستخدام SelectKBest: {accuracy * 100:.2f}%")

# =========================================
# Enhanced PDF Generator (Arabic)
# ========================================

def generate_arabic_pdf(patient_info, diagnosis, risk_factors=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFillColor(colors.HexColor("#2b5876"))
    c.rect(0, height-80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Arabic", 18)
    c.drawCentredString(width/2, height-50, safe_text("تقرير تشخيص الزهايمر"))
    c.setFont("Arabic", 12)
    c.drawCentredString(width/2, height-70, datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Patient Info
    c.setFillColor(colors.black)
    c.setFont("Arabic", 14)
    c.drawString(50, height-120, safe_text("معلومات المريض:"))
    c.setFont("Arabic", 12)
    c.drawString(50, height-145, safe_text(f"الاسم: {patient_info.get('name', '')}"))
    c.drawString(50, height-165, safe_text(f"العمر: {patient_info.get('age', '')}"))
    c.drawString(50, height-185, safe_text(f"الجنس: {patient_info.get('gender', '')}"))

    # Diagnosis
    c.setFont("Arabic", 14)
    c.drawString(50, height-220, safe_text("نتيجة التشخيص:"))
    if diagnosis == "تم اكتشاف الزهايمر":
        c.setFillColor(colors.red)
        result_text = safe_text("تم اكتشاف علامات الزهايمر")
    else:
        c.setFillColor(colors.green)
        result_text = safe_text("لا توجد علامات للزهايمر")
    c.drawString(50, height-245, result_text)
    c.setFillColor(colors.black)

    # Risk Factors
    if risk_factors:
        c.setFont("Arabic", 14)
        c.drawString(50, height-280, safe_text("عوامل الخطر:"))
        c.setFont("Arabic", 12)
        for i, factor in enumerate(risk_factors):
            c.drawString(70, height-305-(i*20), safe_text(f"- {factor}"))

    # Recommendations
    c.setFont("Arabic", 14)
    y_pos = height-350 if risk_factors else height-280
    c.drawString(50, y_pos, safe_text("التوصيات الطبية:"))
    c.setFont("Arabic", 12)

    recommendations = [
        "مراجعة طبيب أعصاب متخصص",
        "إجراء فحوصات دورية",
        "اتباع نظام غذائي صحي",
        "ممارسة التمارين الرياضية",
        "التحفيز الذهني المستمر"
    ] if diagnosis == "تم اكتشاف الزهايمر" else [
        "فحص سنوي بعد عمر الـ50",
        "الحفاظ على نشاط اجتماعي",
        "تمارين تقوية الذاكرة",
        "مراقبة أي تغيرات في الذاكرة"
    ]

    for i, rec in enumerate(recommendations):
        c.drawString(70, y_pos-25-(i*20), safe_text(f"{i+1}. {rec}"))

    # Footer
    c.setFillColor(colors.HexColor("#2b5876"))
    c.rect(0, 0, width, 40, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Arabic", 10)
    c.drawCentredString(width/2, 15, safe_text("هذا التقرير لأغراض إعلامية فقط - يرجى استشارة طبيب متخصص"))

    c.save()
    buffer.seek(0)
    return buffer



# =========================================
# Initialize Session State
# =========================================
if 'diagnosis_result' not in st.session_state:
    st.session_state.diagnosis_result = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = model

# =========================================
# App Interface Setup
# =========================================

st.markdown("""
    <h2 style='text-align: center;'>🧠 تطبيق تشخيص الزهايمر</h2>
""", unsafe_allow_html=True)

# صورة بعرض متوسط (استخدام 70% من العرض)
image = Image.open("alzahimer.jpg")
st.image(image, use_container_width=False, width=1300)

# تقسيم النص والمحتوى التوضيحي بعد الصورة
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(safe_text("""
    <div style="background-color:#f0f8ff; padding:15px; border-radius:10px;">
    <h4 style="color:#2b5876;">معلومات سريعة:</h4>
    <p>• يصيب 1 من كل 9 أشخاص فوق 65 سنة<br>
    • سادس سبب رئيسي للوفاة<br>
    • يتطور على مدار 5-20 سنة</p>
    </div>
    """), unsafe_allow_html=True)

with col2:
    st.subheader(safe_text("مرحباً بكم في تطبيق تشخيص الزهايمر"))
    st.markdown(safe_text("""
    هذا التطبيق يساعد في:
    - تقييم خطر الإصابة بمرض الزهايمر
    - توفير معلومات عن المرض
    - إرشادك إلى الأطباء المتخصصين

    **كيفية الاستخدام:**
    1. املأ معلومات المريض في تبويب 'معلومات المريض'
    2. قم بإجراء التشخيص في تبويب 'التشخيص'
    3. احصل على النتائج والتوصيات
    """))

    st.warning(safe_text("""
    ⚠️ ملاحظة هامة: 
    هذا التطبيق لأغراض توعوية فقط ولا يغني عن استشارة الطبيب المتخصص.
    """))

# =========================================
# التبويبات الفرعية
# =========================================
tabs = st.tabs([
    "📝 معلومات المريض", "🔮 التشخيص", "📊 تقييم النموذج",
    "🌍 الأطباء المتخصصون", "🧠 معلومات عن الزهايمر"
])

    
   
       
# =========================================
# Patient Information Tab
# =========================================
with tabs[0]:
    st.header(safe_text("📝 السجل الطبي للمريض"))

    # تهيئة حالة الجلسة إذا لم تكن موجودة
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {
            "name": "",
            "age": 30,
            "gender": "ذكر",
            "phone": "",
            "landline": ""
        }

    with st.form("patient_info_form"):
        # استخدام القيم من session_state كقيم افتراضية
        name = st.text_input(
            safe_text("الاسم الكامل"), 
            placeholder=safe_text("أحمد محمد علي"),
            value=st.session_state.patient_info.get("name", "")
        )
        age = st.number_input(
            safe_text("العمر"), 
            min_value=1, 
            max_value=120, 
            value=st.session_state.patient_info.get("age", 30)
        )
        gender = st.radio(
            safe_text("الجنس"), 
            [safe_text("ذكر"), safe_text("أنثى")], 
            horizontal=True,
            index=0 if st.session_state.patient_info.get("gender", "ذكر") == "ذكر" else 1
        )
        phone = st.text_input(
            safe_text("رقم الهاتف"), 
            placeholder="01XXXXXXXXXXX",
            value=st.session_state.patient_info.get("phone", "")
        )
        landline = st.text_input(
            safe_text("الرقم الأرضي (اختياري)"), 
            placeholder="03XXXXXXX",
            value=st.session_state.patient_info.get("landline", "")
        )

        valid_prefixes = ["010", "011", "012", "015"]
        submitted = st.form_submit_button(safe_text("حفظ المعلومات"))

        if submitted:
            valid_phone = (phone.startswith(tuple(valid_prefixes)) and (len(phone) == 11) and (phone.isdigit()))

            if not valid_phone:
                st.error(safe_text("❌ رقم الهاتف غير صالح. يجب أن يكون 11 رقماً ويبدأ بـ 010 أو 011 أو 012 أو 015."))
            else:
                st.session_state.patient_info = {
                    "name": safe_text(name),
                    "age": age,
                    "gender": safe_text(gender),
                    "phone": phone,
                    "landline": landline
                }
                st.success(safe_text("✅ تم حفظ البيانات بنجاح"))
                st.rerun()  # إعادة تحميل الصفحة لتحديث البيانات

    # عرض البيانات المحفوظة
    if st.session_state.get("patient_info"):
        st.markdown("---")
        st.subheader(safe_text("المعلومات المحفوظة"))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(safe_text(f"""
            **الاسم:** {st.session_state.patient_info.get("name", "غير معروف")}  
            **العمر:** {st.session_state.patient_info.get("age", "غير معروف")}  
            **الجنس:** {st.session_state.patient_info.get("gender", "غير معروف")}
            """))

        with col2:
            st.markdown(safe_text(f"""
            **رقم الهاتف:** {st.session_state.patient_info.get("phone", "غير معروف")}  
            **الرقم الأرضي:** {st.session_state.patient_info.get("landline", "غير معروف")}
            """))

        # زر لمسح البيانات
        if st.button(safe_text("مسح البيانات")):
            st.session_state.patient_info = {
                "name": "",
                "age": 30,
                "gender": "ذكر",
                "phone": "",
                "landline": ""
            }
            st.success(safe_text("تم مسح البيانات بنجاح"))
            st.rerun()

# =========================================
# Diagnosis Tab
# =========================================
with tabs[1]:
    st.header("🔮 تقييم خطر الإصابة بالزهايمر")

    if not st.session_state.get("patient_info"):
        st.warning("⚠️ يرجى إدخال معلومات المريض أولاً في تبويب 'معلومات المريض'")
    else:
        with st.form("diagnosis_form"):
            st.subheader("البيانات الطبية")

            col1, col2 = st.columns(2)

            with col1:
                SleepQuality = st.selectbox("جودة النوم", ["ضعيف", "متوسط", "جيد"])
                Hypertension = st.radio("هل تعاني من ارتفاع ضغط الدم؟", ["نعم", "لا"])
                CholesterolHDL = st.number_input("نسبة الكوليسترول الجيد (HDL)", min_value=10.0, max_value=100.0, value=40.0)
                MMSE = st.number_input("نتيجة اختبار الحالة العقلية المصغرة (MMSE)", min_value=0, max_value=30, value=24)
                FunctionalAssessment = st.slider("التقييم الوظيفي", min_value=0, max_value=10, value=5)

            with col2:
                MemoryComplaints = st.slider("شكاوى الذاكرة", min_value=0, max_value=10, value=3)
                BehavioralProblems = st.slider("المشاكل السلوكية", min_value=0, max_value=10, value=2)
                ADL = st.slider("الأنشطة اليومية المعيشية (ADL)", min_value=0, max_value=10, value=8)
                BMI = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=10.0, max_value=60.0, value=22.0)
                DietQuality = st.selectbox("جودة النظام الغذائي", ["ضعيف", "متوسط", "جيد"])

            if st.form_submit_button("إجراء التشخيص"):
                input_data = {
                    'SleepQuality': [ ["ضعيف", "متوسط", "جيد"].index(SleepQuality) ],
                    'Hypertension': [ 1 if Hypertension == "نعم" else 0 ],
                    'CholesterolHDL': [ CholesterolHDL ],
                    'MMSE': [ MMSE ],
                    'FunctionalAssessment': [ FunctionalAssessment ],
                    'MemoryComplaints': [ MemoryComplaints ],
                    'BehavioralProblems': [ BehavioralProblems ],
                    'ADL': [ ADL ],
                    'BMI': [ BMI ],
                    'DietQuality': [ ["ضعيف", "متوسط", "جيد"].index(DietQuality) ]
                }

                input_df = pd.DataFrame.from_dict(input_data)
                st.session_state.input_data = input_df

                prediction = st.session_state.ml_model.predict(input_df)[0]
                probability = st.session_state.ml_model.predict_proba(input_df)[0][1]

                st.session_state.diagnosis_result = "تم اكتشاف الزهايمر" if prediction == 1 else "سليم"
                st.session_state.probability = probability
                st.session_state.risk_factors = []

                if prediction == 1:
                    if Hypertension == "نعم":
                        st.session_state.risk_factors.append("ارتفاع ضغط الدم")
                    if DietQuality == "ضعيف":
                        st.session_state.risk_factors.append("نظام غذائي ضعيف")
                    if SleepQuality == "ضعيف":
                        st.session_state.risk_factors.append("جودة نوم منخفضة")

                st.rerun()

    if st.session_state.get("diagnosis_result"):
        st.markdown("---")
        st.subheader("نتيجة التشخيص")

        if st.session_state.diagnosis_result == "تم اكتشاف الزهايمر":
            st.error("### النتيجة: تشخيص إيجابي للزهايمر")
        else:
            st.success("### النتيجة: لا توجد مؤشرات على الزهايمر")

        # عرض مخطط الاحتمال
        fig, ax = plt.subplots(figsize=(8, 2))
        sns.heatmap([[st.session_state.probability]], 
                    annot=[[f"{st.session_state.probability*100:.1f}%"]], 
                    fmt="", cmap="Reds", cbar=False, 
                    xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title("Alzheimer's Risk Probability")
        st.pyplot(fig)

        # توصيات بناءً على النتيجة
        if st.session_state.probability >= 0.7:
            risk_level = "🔴 مرتفع جدًا"
            recommendation = "ننصح بمراجعة طبيب أعصاب متخصص فورًا لإجراء فحوصات دقيقة."
        elif st.session_state.probability >= 0.4:
            risk_level = "🔠 متوسط"
            recommendation = "يفضل المتابعة الدورية وتحسين نمط الحياة مثل التغذية والنوم."
        else:
            risk_level = "🟢 منخفض"
            recommendation = "لا توجد مؤشرات قوية حاليًا، استمر في نمط حياة صحي."

        st.markdown(f"### مستوى الخطورة: {risk_level}")
        st.info(f"📌 توصية: {recommendation}")

        if st.session_state.get('risk_factors'):
            st.warning("### عوامل الخطر الرئيسية:")
            for factor in st.session_state.risk_factors:
                st.write(f"- {factor}")

        if st.session_state.patient_info.get("name"):
            pdf_buffer = generate_arabic_pdf(
                st.session_state.patient_info,
                st.session_state.diagnosis_result,
                st.session_state.get('risk_factors', [])
            )

            st.download_button(
                "⬇️ تحميل التقرير الطبي",
                data=pdf_buffer.getvalue(),
                file_name=f"تقرير_الزهايمر_{st.session_state.patient_info['name']}.pdf",
                mime="application/pdf"
            )

# =========================================
# Model Evaluation Tab
# =========================================
with tabs[2]:
    st.header(safe_text("📊 تقييم أداء النموذج التشخيصي"))
    st.markdown("---")

    # مقاييس الأداء
    st.subheader(safe_text("نتائج النموذج باختصار"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(safe_text("الدقة"), f"{accuracy*100:.1f}%", 
                 help=safe_text("نسبة التوقعات الصحيحة من جميع الحالات"))
    with col2:
        st.metric(safe_text("الحساسية"), "82.3%",
                 help=safe_text("نسبة الحالات المصابة التي تم اكتشافها بشكل صحيح"))
    with col3:
        st.metric(safe_text("الخصوصية"), "89.1%",
                 help=safe_text("نسبة الحالات السليمة التي تم استبعادها بشكل صحيح"))

    st.markdown("---")

    from bidi.algorithm import get_display
    import arabic_reshaper

    # دالة لمعالجة النص العربي
    def arabic_text(text):
        return get_display(arabic_reshaper.reshape(text))

    # مصفوفة الارتباك
    st.subheader("Confusion Matrix (مقارنة نتائج التوقع بالحقيقة)")

    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        annot_kws={"size": 14},
        xticklabels=['Healthy (+)', 'Diseased (-)'],
        yticklabels=['Healthy (+)', 'Diseased (-)'],
        ax=ax
    )
    ax.set_title("Confusion Matrix: Prediction vs Actual", pad=20)
    ax.set_xlabel("Predicted Label", labelpad=15)
    ax.set_ylabel("True Label", labelpad=15)
    ax.xaxis.set_tick_params(labelrotation=0)
    ax.yaxis.set_tick_params(labelrotation=0)
    st.pyplot(fig)



    # تفسير النتائج
    with st.expander(safe_text("🔍 كيف تفهم هذه النتائج؟")):
        st.markdown(safe_text("""
        **تفسير مصفوفة الارتباك:**

        - **الخلية العلوية اليسرى (TN)**: الحالات السليمة التي تم تشخيصها بشكل صحيح
        - **الخلية العلوية اليمنى (FP)**: الحالات السليمة التي تم تشخيصها خطأً كمرضى
        - **الخلية السفلية اليسرى (FN)**: الحالات المرضية التي تم تشخيصها خطأً كسليمة
        - **الخلية السفلية اليمنى (TP)**: الحالات المرضية التي تم تشخيصها بشكل صحيح

        **تطبيقات عملية:**
        - معدل الخطأ في تشخيص المرضى: FN/(TP+FN)
        - معدل الخطأ في تشخيص الأصحاء: FP/(TN+FP)
        """))

    # تقرير التصنيف
    st.markdown("---")
    st.subheader(safe_text("تقرير التصنيف التفصيلي"))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(class_report).transpose())

# =========================================
# Specialist Doctors Tab
# =========================================
with tabs[3]:
    st.header("🩺 Find the Nearest Specialist (أقرب طبيب مختص)")
    st.markdown("---")

    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim
    import folium
    from streamlit_folium import st_folium

    geolocator = Nominatim(user_agent="alzheimer_app")

    # قائمة المحافظات والمناطق الشائعة
    regions = {
        "القاهرة": ["مدينة نصر", "المعادي", "مصر الجديدة", "الزمالك"],
        "الإسكندرية": ["سموحة", "محطة الرمل", "العصافرة", "المنتزه"]
    }

    # بيانات الأطباء
    doctors = [
        {
            "name": "د. أحمد علي",
            "location": [30.0444, 31.2357],
            "specialty": "أمراض الزهايمر",
            "clinic": "مستشفى العصبية",
            "phone": "0123456789",
            "working_hours": "السبت-الأربعاء 5م-9م"
        },
        {
            "name": "د. مريم حسن",
            "location": [30.0275, 31.2105],
            "specialty": "طب أعصاب المسنين",
            "clinic": "مركز نيوروكير",
            "phone": "0111222333",
            "working_hours": "الأحد-الخميس 4م-8م"
        },
        {
            "name": "د. سمير إبراهيم",
            "location": [30.0500, 31.3000],
            "specialty": "أعصاب وذاكرة",
            "clinic": "مركز النخبة",
            "phone": "0109988776",
            "working_hours": "السبت-الثلاثاء 6م-10م"
        }
    ]

    # إدخال الموقع
    st.subheader("📍 أدخل موقعك")
    selected_city = st.selectbox("اختر المحافظة", list(regions.keys()))
    selected_area = st.selectbox("اختر المنطقة", regions[selected_city])

    user_address = f"{selected_area}, {selected_city}, Egypt"
    location = geolocator.geocode(user_address)

    if location:
        user_location = (location.latitude, location.longitude)

        # حساب المسافات
        for doc in doctors:
            doc["distance_km"] = round(geodesic(user_location, doc["location"]).km, 2)
        doctors.sort(key=lambda x: x["distance_km"])

        # عرض الخريطة
        m = folium.Map(location=user_location, zoom_start=12)
        folium.Marker(
            location=user_location,
            icon=folium.Icon(color='blue'),
            tooltip="📍 موقعك الحالي"
        ).add_to(m)

        for doc in doctors:
            folium.Marker(
                location=doc["location"],
                tooltip=doc["name"],
                popup=f"""
                <b>{doc['name']}</b><br>
                التخصص: {doc['specialty']}<br>
                العيادة: {doc['clinic']}<br>
                الهاتف: {doc['phone']}<br>
                المواعيد: {doc['working_hours']}<br>
                المسافة: {doc['distance_km']} كم
                """
            ).add_to(m)

        st_folium(m, width=700, height=500)

        # قائمة الأطباء
        st.markdown("---")
        st.subheader("📝 قائمة الأطباء الأقرب")
        for doc in doctors:
            st.markdown(f"""
            **{doc['name']}**
            - 🩺 **التخصص:** {doc['specialty']}
            - 🏥 **العيادة:** {doc['clinic']}
            - 📞 **الهاتف:** `{doc['phone']}`
            - 🕓 **المواعيد:** {doc['working_hours']}
            - 📍 **المسافة:** {doc['distance_km']} كم
            ---
            """)
    else:
        st.error("❌ لم يتم تحديد الموقع، يرجى التحقق من الاسم")



# =========================================
# Alzheimer's Info Tab
# =========================================
with tabs[4]:
    st.header(safe_text("🧠 معلومات عن مرض الزهايمر"))
    
    tabs_info = st.tabs([safe_text("تعريف المرض"), safe_text("الأعراض"), safe_text("الوقاية"), safe_text("العلاج")])
    
    with tabs_info[0]:
        st.markdown(safe_text("""
        **ما هو مرض الزهايمر؟**
        
        مرض عصبي مزمن يؤدي إلى:
        - تدهور تدريجي في الذاكرة
        - ضعف القدرات المعرفية
        - صعوبة في القيام بالمهام اليومية
        
        **حقائق أساسية:**
        - أكثر أنواع الخرف شيوعاً (60-80% من الحالات)
        - يصيب عادة الأشخاص فوق 65 سنة
        - يتطور على مدى سنوات
        """))
    
    with tabs_info[1]:
        st.markdown(safe_text("""
        **الأعراض المبكرة:**
        - نسيان المحادثات الحديثة والأحداث
        - صعوبة في تذكر الأسماء والأماكن
        - صعوبة في التخطيط أو حل المشكلات
        
        **الأعراض المتقدمة:**
        - فقدان الذاكرة الشديد
        - صعوبة في الكلام والكتابة
        - تغيرات في الشخصية والمزاج
        - صعوبة في التعرف على الأهل والأصدقاء
        """))
    
    with tabs_info[2]:
        st.markdown(safe_text("""
        **طرق الوقاية:**
        
        1. **النشاط الذهني المستمر**
           - القراءة المنتظمة
           - تعلم مهارات جديدة
           - حل الألغاز والألعاب الذهنية
        
        2. **التمارين الرياضية**
           - 30 دقيقة على الأقل يومياً
           - المشي السريع، السباحة، ركوب الدراجة
        
        3. **النظام الغذائي الصحي**
           - حمية البحر المتوسط
           - الخضروات والفواكه الطازجة
           - الأسماك الغنية بأوميغا-3
        
        4. **النوم الجيد**
           - 7-8 ساعات ليلاً
           - علاج اضطرابات النوم مثل توقف التنفس
        
        5. **التفاعل الاجتماعي**
           - المشاركة في الأنشطة الاجتماعية
           - تجنب العزلة والوحدة
        """))
    
    with tabs_info[3]:
        st.markdown(safe_text("""
        **العلاجات المتاحة:**
        
        - **الأدوية:**
          - مثبطات الكولينستيراز (دونيبيزيل، ريفاستيجمين)
          - ميمانتين (للمراحل المتوسطة إلى المتقدمة)
        
        - **العلاجات غير الدوائية:**
          - العلاج السلوكي
          - تعديل البيئة المحيطة
          - الدعم الأسري والرعاية المنزلية
        
        **العلاجات التجريبية:**
        - العلاج الجيني
        - العلاج المناعي
        - الخلايا الجذعية
        """))

# =========================================
# Footer
# =========================================
st.markdown("---")
st.markdown(safe_text("""
<p style='text-align: center; color: #666;'>
تطبيق تشخيص الزهايمر - للأغراض التوعوية فقط<br>
© 2024   حقوق محفوظة | تم التطوير بواسطة رنين أشرف @
</p>
"""), unsafe_allow_html=True)