import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Jaya Jaya Institut — Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
 
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
 
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #db2777 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .main-header h1 { font-size: 2rem; font-weight: 800; color: white; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1rem; }
 
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
 
    .result-dropout  { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 2px solid #ef4444; border-radius: 16px; padding: 2rem; text-align: center; }
    .result-graduate { background: linear-gradient(135deg, #052e16, #14532d); border: 2px solid #22c55e; border-radius: 16px; padding: 2rem; text-align: center; }
    .result-enrolled { background: linear-gradient(135deg, #172554, #1e3a8a); border: 2px solid #3b82f6; border-radius: 16px; padding: 2rem; text-align: center; }
    .result-title  { font-size: 1.2rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.5rem; }
    .result-status { font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; }
 
    .risk-high   { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; padding: 0.3rem 1rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
    .risk-medium { background: #713f12; color: #fcd34d; border: 1px solid #f59e0b; padding: 0.3rem 1rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
    .risk-low    { background: #052e16; color: #86efac; border: 1px solid #22c55e; padding: 0.3rem 1rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
 
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #60a5fa;
        text-transform: uppercase; letter-spacing: 2px;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #7c3aed) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-size: 1rem !important; padding: 0.75rem 2rem !important;
        width: 100% !important;
    }
    .info-box {
        background: #172554; border: 1px solid #3b82f6;
        border-radius: 10px; padding: 1rem 1.25rem;
        margin: 1rem 0; font-size: 0.9rem; color: #bfdbfe;
    }
    .warning-box {
        background: #451a03; border: 1px solid #f59e0b;
        border-radius: 10px; padding: 1rem 1.25rem;
        margin: 1rem 0; font-size: 0.9rem; color: #fde68a;
    }
    hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)
 
 
# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model   = joblib.load("model/best_model.pkl")
    scaler  = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    return model, scaler, encoder
 
model, scaler, le_target = load_model()
 
 
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_input(df_input):
    categorical_cols = [
        'Marital_status', 'Application_mode', 'Course',
        'Daytime_evening_attendance', 'Previous_qualification',
        'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
        'Mothers_occupation', 'Fathers_occupation',
        'Displaced', 'Educational_special_needs', 'Debtor',
        'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International'
    ]
    binary_cols = [
        'Daytime_evening_attendance', 'Displaced', 'Educational_special_needs',
        'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
        'International', 'Financial_Risk'
    ]
 
    X = df_input.copy()
 
    # Feature Engineering
    X['Approval_Rate_Sem1'] = X['Curricular_units_1st_sem_approved'] / X['Curricular_units_1st_sem_enrolled'].replace(0, 1)
    X.loc[X['Curricular_units_1st_sem_enrolled'] == 0, 'Approval_Rate_Sem1'] = 0
    X['Approval_Rate_Sem2'] = X['Curricular_units_2nd_sem_approved'] / X['Curricular_units_2nd_sem_enrolled'].replace(0, 1)
    X.loc[X['Curricular_units_2nd_sem_enrolled'] == 0, 'Approval_Rate_Sem2'] = 0
    X['Grade_Per_Unit_Sem1'] = X['Curricular_units_1st_sem_grade'] / X['Curricular_units_1st_sem_enrolled'].replace(0, 1)
    X.loc[X['Curricular_units_1st_sem_enrolled'] == 0, 'Grade_Per_Unit_Sem1'] = 0
    X['Grade_Per_Unit_Sem2'] = X['Curricular_units_2nd_sem_grade'] / X['Curricular_units_2nd_sem_enrolled'].replace(0, 1)
    X.loc[X['Curricular_units_2nd_sem_enrolled'] == 0, 'Grade_Per_Unit_Sem2'] = 0
    X['Financial_Risk'] = ((X['Debtor'] == 1) | (X['Tuition_fees_up_to_date'] == 0)).astype(int)
    X['Grade_Change_Sem1_Sem2'] = X['Curricular_units_2nd_sem_grade'] - X['Curricular_units_1st_sem_grade']
 
    categorical_cols = categorical_cols + ['Financial_Risk']
    numerical_cols   = [col for col in X.columns if col not in categorical_cols]
    ohe_cols         = [col for col in categorical_cols if col not in binary_cols]
 
    # Capping (untuk single row gunakan nilai tetap)
    for col in numerical_cols:
        if len(X) > 1:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            X[col] = X[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
 
    # OHE
    X_enc = pd.get_dummies(X, columns=ohe_cols, drop_first=False)
 
    # Scale
    scale_cols = [c for c in numerical_cols if c in X_enc.columns]
    X_enc[scale_cols] = scaler.transform(X_enc[scale_cols])
 
    # Align kolom
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_enc.columns
    X_enc = X_enc.reindex(columns=feature_names, fill_value=0)
    return X_enc
 
 
def predict(X_processed):
    proba = model.predict_proba(X_processed)
    pred  = model.predict(X_processed)
    return le_target.inverse_transform(pred), proba
 
 
def get_risk(dropout_pct):
    if dropout_pct >= 60:   return "High Risk",   "risk-high"
    elif dropout_pct >= 30: return "Medium Risk",  "risk-medium"
    else:                   return "Low Risk",     "risk-low"
 
 
# Mapping options
MARITAL_MAP   = {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Facto Union",6:"Legally Separated"}
COURSE_MAP    = {33:"Biofuel Production",171:"Animation & Multimedia",8014:"Social Service (Eve)",
                 9003:"Agronomy",9070:"Communication Design",9085:"Veterinary Nursing",
                 9119:"Informatics Engineering",9130:"Equinculture",9147:"Management",
                 9238:"Social Service",9254:"Tourism",9500:"Nursing",9556:"Oral Hygiene",
                 9670:"Advertising & Marketing",9773:"Journalism",9853:"Basic Education",9991:"Management (Eve)"}
PREV_QUAL_MAP = {1:"Secondary Education",2:"Bachelor's",3:"Degree",4:"Master's",5:"Doctorate",
                 6:"Higher Ed Frequency",9:"12th Not Complete",10:"11th Not Complete",12:"Other 11th",
                 14:"10th Year",15:"10th Not Complete",19:"Basic 3rd Cycle",38:"Basic 2nd Cycle",
                 39:"Tech Specialization",40:"Degree 1st Cycle",42:"Prof Higher Tech",43:"Master's 2nd Cycle"}
NATION_MAP    = {1:"Portuguese",2:"German",6:"Spanish",11:"Italian",13:"Dutch",14:"English",
                 17:"Lithuanian",21:"Angolan",22:"Cape Verdean",24:"Guinean",25:"Mozambican",
                 26:"Santomean",32:"Turkish",41:"Brazilian",62:"Romanian",100:"Moldovan",
                 101:"Mexican",103:"Ukrainian",105:"Russian",108:"Cuban",109:"Colombian"}
QUAL_MAP      = {1:"Secondary",2:"Bachelor's",3:"Degree",4:"Master's",5:"Doctorate",6:"Higher Ed Freq",
                 9:"12th Not Complete",10:"11th Not Complete",11:"7th Year Old",12:"Other 11th",
                 14:"10th Year",18:"Commerce Course",19:"Basic 3rd Cycle",22:"Tech Professional",
                 26:"7th Year",27:"2nd Cycle HS",29:"9th Not Complete",30:"8th Year",34:"Unknown",
                 35:"Cannot Read/Write",36:"Can Read No 4th",37:"Basic 1st Cycle",38:"Basic 2nd Cycle",
                 39:"Tech Specialization",40:"Degree 1st Cycle",41:"Specialized Higher",
                 42:"Prof Higher Tech",43:"Master's 2nd",44:"Doctorate 3rd"}
OCC_MAP       = {0:"Student",1:"Legislative/Exec",2:"Intellectual/Sci",3:"Technicians",4:"Admin Staff",
                 5:"Personal Services",6:"Agriculture",7:"Industry/Construction",8:"Machine Operators",
                 9:"Unskilled",10:"Armed Forces",90:"Other",99:"Unknown",122:"Health Professionals",
                 123:"Teachers",125:"ICT Specialists",131:"Science/Eng Tech",132:"Health Tech",
                 134:"Legal/Social Tech",141:"Office Workers",143:"Accounting",144:"Other Admin",
                 151:"Personal Service",152:"Sellers",153:"Personal Care",171:"Construction",
                 173:"Printing",175:"Food Processing",191:"Cleaning",192:"Agriculture Workers",
                 193:"Industry Workers",194:"Meal Preparation"}
APP_MODE_MAP  = {1:"1st Phase General",2:"Ordinance 612/93",5:"Special (Azores)",7:"Other Higher Courses",
                 10:"Ordinance 854-B/99",15:"International Bachelor",16:"Special (Madeira)",
                 17:"2nd Phase General",18:"3rd Phase General",26:"Ord 533-A b2",27:"Ord 533-A b3",
                 39:"Over 23 Years",42:"Transfer",43:"Change of Course",44:"Tech Specialization",
                 51:"Change Institution",53:"Short Cycle",57:"Change Intl"}
 
 
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🎓</div>
        <div style='font-size:1.1rem; font-weight:800; color:#e2e8f0;'>Jaya Jaya Institut</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:0.3rem;'>Dropout Early Warning System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.85rem; color:#94a3b8; line-height:1.8;'>
    <b style='color:#60a5fa;'>📌 Tentang Aplikasi</b><br>
    Prediksi risiko dropout mahasiswa menggunakan Machine Learning.<br><br>
    <b style='color:#60a5fa;'>🤖 Model</b><br>
    RandomForestClassifier<br>
    Accuracy: 76.95%<br><br>
    <b style='color:#60a5fa;'>📊 Kelas Prediksi</b><br>
    🔴 Dropout &nbsp;&nbsp; 🔵 Enrolled &nbsp;&nbsp; 🟢 Graduate<br><br>
    <b style='color:#60a5fa;'>⚠️ Risk Category</b><br>
    🔴 High Risk : ≥ 60%<br>
    🟡 Medium Risk: 30–60%<br>
    🟢 Low Risk  : &lt; 30%
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#475569; text-align:center;'>Jaya Jaya Institut © 2026<br>Data Science Division</div>", unsafe_allow_html=True)
 
 
# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class='main-header'>
    <h1>🎓 Student Dropout Predictor</h1>
    <p>Sistem deteksi dini risiko dropout mahasiswa berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)
 
tab1, tab2 = st.tabs(["📋  Prediksi Individual", "📂  Prediksi Batch (CSV)"])
 
 
# ============================================================
# TAB 1 — PREDIKSI INDIVIDUAL
# ============================================================
with tab1:
    with st.form("prediction_form"):
 
        st.markdown("<div class='section-header'>👤 Data Demografis</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender         = st.selectbox("Gender", options=list({1:"Male",0:"Female"}.keys()), format_func=lambda x: {1:"Male",0:"Female"}[x])
            age            = st.number_input("Usia saat Pendaftaran", min_value=17, max_value=70, value=20)
            marital_status = st.selectbox("Status Pernikahan", options=list(MARITAL_MAP.keys()), format_func=lambda x: MARITAL_MAP[x])
        with c2:
            nationality    = st.selectbox("Kewarganegaraan", options=list(NATION_MAP.keys()), format_func=lambda x: NATION_MAP[x])
            displaced      = st.selectbox("Displaced", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
            international  = st.selectbox("International Student", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
        with c3:
            special_needs    = st.selectbox("Special Needs", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
            application_mode = st.selectbox("Application Mode", options=list(APP_MODE_MAP.keys()), format_func=lambda x: APP_MODE_MAP[x])
            application_order= st.number_input("Application Order (0-9)", min_value=0, max_value=9, value=1)
 
        st.markdown("<div class='section-header'>📚 Data Akademik</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            course           = st.selectbox("Program Studi", options=list(COURSE_MAP.keys()), format_func=lambda x: COURSE_MAP[x])
            attendance       = st.selectbox("Waktu Kuliah", options=[1,0], format_func=lambda x: "Daytime" if x==1 else "Evening")
            prev_qualification= st.selectbox("Kualifikasi Sebelumnya", options=list(PREV_QUAL_MAP.keys()), format_func=lambda x: PREV_QUAL_MAP[x])
            prev_grade       = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", min_value=0.0, max_value=200.0, value=120.0)
            admission_grade  = st.number_input("Nilai Masuk (0-200)", min_value=0.0, max_value=200.0, value=130.0)
        with c2:
            st.markdown("**Semester 1**")
            cu1_credited    = st.number_input("Credited Sem 1",     min_value=0, max_value=20, value=0)
            cu1_enrolled    = st.number_input("Enrolled Sem 1",     min_value=0, max_value=20, value=6)
            cu1_evaluations = st.number_input("Evaluations Sem 1",  min_value=0, max_value=40, value=6)
            cu1_approved    = st.number_input("Approved Sem 1",     min_value=0, max_value=20, value=6)
            cu1_grade       = st.number_input("Grade Sem 1 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
            cu1_no_eval     = st.number_input("Without Eval Sem 1", min_value=0, max_value=20, value=0)
        with c3:
            st.markdown("**Semester 2**")
            cu2_credited    = st.number_input("Credited Sem 2",     min_value=0, max_value=20, value=0)
            cu2_enrolled    = st.number_input("Enrolled Sem 2",     min_value=0, max_value=20, value=6)
            cu2_evaluations = st.number_input("Evaluations Sem 2",  min_value=0, max_value=40, value=6)
            cu2_approved    = st.number_input("Approved Sem 2",     min_value=0, max_value=20, value=6)
            cu2_grade       = st.number_input("Grade Sem 2 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
            cu2_no_eval     = st.number_input("Without Eval Sem 2", min_value=0, max_value=20, value=0)
 
        st.markdown("<div class='section-header'>💰 Data Keuangan & Keluarga</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            scholarship = st.selectbox("Penerima Beasiswa", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
            debtor      = st.selectbox("Memiliki Hutang",   options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
            tuition_ok  = st.selectbox("SPP Tepat Waktu",   options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
        with c2:
            mothers_qual = st.selectbox("Pendidikan Ibu", options=list(QUAL_MAP.keys()), format_func=lambda x: QUAL_MAP[x])
            mothers_occ  = st.selectbox("Pekerjaan Ibu",  options=list(OCC_MAP.keys()),  format_func=lambda x: OCC_MAP[x])
        with c3:
            fathers_qual = st.selectbox("Pendidikan Ayah", options=list(QUAL_MAP.keys()), format_func=lambda x: QUAL_MAP[x])
            fathers_occ  = st.selectbox("Pekerjaan Ayah",  options=list(OCC_MAP.keys()),  format_func=lambda x: OCC_MAP[x])
 
        st.markdown("<div class='section-header'>🌐 Faktor Ekonomi Makro</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0,  max_value=25.0, value=10.8)
        with c2: inflation    = st.number_input("Inflation Rate (%)",    min_value=-5.0, max_value=15.0, value=1.4)
        with c3: gdp          = st.number_input("GDP",                   min_value=-5.0, max_value=5.0,  value=1.74)
 
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Prediksi Sekarang")
 
    if submitted:
        input_data = pd.DataFrame([{
            'Marital_status': marital_status, 'Application_mode': application_mode,
            'Application_order': application_order, 'Course': course,
            'Daytime_evening_attendance': attendance, 'Previous_qualification': prev_qualification,
            'Previous_qualification_grade': prev_grade, 'Nacionality': nationality,
            'Mothers_qualification': mothers_qual, 'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ, 'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade, 'Displaced': displaced,
            'Educational_special_needs': special_needs, 'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_ok, 'Gender': gender,
            'Scholarship_holder': scholarship, 'Age_at_enrollment': age, 'International': international,
            'Curricular_units_1st_sem_credited': cu1_credited, 'Curricular_units_1st_sem_enrolled': cu1_enrolled,
            'Curricular_units_1st_sem_evaluations': cu1_evaluations, 'Curricular_units_1st_sem_approved': cu1_approved,
            'Curricular_units_1st_sem_grade': cu1_grade, 'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
            'Curricular_units_2nd_sem_credited': cu2_credited, 'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
            'Curricular_units_2nd_sem_evaluations': cu2_evaluations, 'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade, 'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
            'Unemployment_rate': unemployment, 'Inflation_rate': inflation, 'GDP': gdp
        }])
 
        X_proc = preprocess_input(input_data)
        pred_status, proba = predict(X_proc)
 
        dropout_pct  = round(proba[0][0] * 100, 1)
        enrolled_pct = round(proba[0][1] * 100, 1)
        graduate_pct = round(proba[0][2] * 100, 1)
        risk_label, risk_class = get_risk(dropout_pct)
        status = pred_status[0]
 
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")
 
        card_class = {"Dropout":"result-dropout","Enrolled":"result-enrolled","Graduate":"result-graduate"}.get(status,"result-enrolled")
        emoji      = {"Dropout":"🔴","Enrolled":"🔵","Graduate":"🟢"}.get(status,"⚪")
 
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='{card_class}'>
                <div class='result-title'>Status Prediksi</div>
                <div class='result-status'>{emoji} {status}</div>
                <div style='margin-top:0.5rem;'><span class='{risk_class}'>{risk_label}</span></div>
                <div style='color:#94a3b8; font-size:0.85rem; margin-top:0.75rem;'>Risk Score: {dropout_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("**Probabilitas per Kelas:**")
            st.progress(dropout_pct / 100);  st.caption(f"🔴 Dropout  : {dropout_pct}%")
            st.progress(enrolled_pct / 100); st.caption(f"🔵 Enrolled : {enrolled_pct}%")
            st.progress(graduate_pct / 100); st.caption(f"🟢 Graduate : {graduate_pct}%")
 
        st.markdown("<br>", unsafe_allow_html=True)
        if status == "Dropout" or risk_label == "High Risk":
            st.markdown("""<div class='warning-box'>⚠️ <b>Rekomendasi Tindakan:</b><br>
            Mahasiswa ini berisiko tinggi dropout. Segera lakukan:<br>
            • Konseling akademik dan evaluasi beban studi<br>
            • Cek status keuangan dan informasikan program beasiswa<br>
            • Bimbingan khusus dari dosen wali<br>
            • Monitoring progress akademik per minggu</div>""", unsafe_allow_html=True)
        elif status == "Graduate":
            st.markdown("""<div class='info-box'>✅ <b>Mahasiswa berpotensi lulus!</b><br>
            Pertahankan performa akademik dan dukungan yang ada.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='info-box'>📊 <b>Status Enrolled — Perlu Monitoring Lanjutan</b><br>
            Pantau perkembangan akademik dan faktor keuangan secara berkala.</div>""", unsafe_allow_html=True)
 
 
# ============================================================
# TAB 2 — PREDIKSI BATCH
# ============================================================
with tab2:
    st.markdown("""<div class='info-box'>
    📂 <b>Upload file CSV</b> dengan format kolom yang sama seperti dataset asli.<br>
    Separator bisa menggunakan titik koma <code>;</code> atau koma <code>,</code>.<br>
    Kolom <code>Status</code> tidak wajib ada.
    </div>""", unsafe_allow_html=True)
 
    template_cols = [
        'Marital_status','Application_mode','Application_order','Course',
        'Daytime_evening_attendance','Previous_qualification','Previous_qualification_grade',
        'Nacionality','Mothers_qualification','Fathers_qualification',
        'Mothers_occupation','Fathers_occupation','Admission_grade',
        'Displaced','Educational_special_needs','Debtor','Tuition_fees_up_to_date',
        'Gender','Scholarship_holder','Age_at_enrollment','International',
        'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade','Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited','Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations','Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade','Curricular_units_2nd_sem_without_evaluations',
        'Unemployment_rate','Inflation_rate','GDP'
    ]
    csv_template = pd.DataFrame(columns=template_cols).to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("⬇️ Download Template CSV", data=csv_template, file_name="template_mahasiswa.csv", mime="text/csv")
 
    uploaded_file = st.file_uploader("Upload CSV Mahasiswa", type=["csv"])
 
    if uploaded_file is not None:
        try:
            try:
                df_upload = pd.read_csv(uploaded_file, sep=';')
                if df_upload.shape[1] < 5:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, sep=',')
            except:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=',')
 
            st.success(f"✅ {len(df_upload)} data mahasiswa berhasil diupload")
            st.dataframe(df_upload.head(3), use_container_width=True)
 
            if st.button("🚀 Jalankan Prediksi Batch"):
                with st.spinner("Memproses prediksi..."):
                    df_pred   = df_upload.drop(columns=['Status'], errors='ignore').copy()
                    X_proc    = preprocess_input(df_pred)
                    pred_status, proba = predict(X_proc)
                    dropout_pct_arr = (proba[:, 0] * 100).round(1)
 
                    df_result = df_upload.copy()
                    df_result['Predicted_Status']  = pred_status
                    df_result['Risk_Score_Dropout'] = dropout_pct_arr
                    df_result['Risk_Category']      = pd.cut(
                        dropout_pct_arr, bins=[0, 30, 60, 100],
                        labels=['Low Risk', 'Medium Risk', 'High Risk']
                    ).astype(str)
 
                st.markdown("### 📊 Ringkasan Hasil Prediksi")
                counts      = pd.Series(pred_status).value_counts()
                risk_counts = df_result['Risk_Category'].value_counts()
 
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"<div class='metric-card'><div class='value' style='color:#f87171;'>{counts.get('Dropout',0)}</div><div class='label'>Dropout</div></div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='metric-card'><div class='value' style='color:#60a5fa;'>{counts.get('Enrolled',0)}</div><div class='label'>Enrolled</div></div>", unsafe_allow_html=True)
                with c3: st.markdown(f"<div class='metric-card'><div class='value' style='color:#4ade80;'>{counts.get('Graduate',0)}</div><div class='label'>Graduate</div></div>", unsafe_allow_html=True)
                with c4: st.markdown(f"<div class='metric-card'><div class='value' style='color:#fb923c;'>{risk_counts.get('High Risk',0)}</div><div class='label'>High Risk</div></div>", unsafe_allow_html=True)
 
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Detail Hasil Prediksi")
                display_cols = (['Status'] if 'Status' in df_upload.columns else []) + ['Predicted_Status','Risk_Score_Dropout','Risk_Category']
                st.dataframe(df_result[display_cols], use_container_width=True)
 
                csv_result = df_result.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button("⬇️ Download Hasil Prediksi CSV", data=csv_result, file_name="hasil_prediksi.csv", mime="text/csv")
 
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")
            st.info("Pastikan file CSV menggunakan separator titik koma (;) atau koma (,)")