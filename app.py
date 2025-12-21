import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
    🎓 College Placement Prediction System
    </h1>
    <p style='text-align:center; font-size:18px;'>
    AI-powered system to predict student placement status
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUT SECTIONS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Student Profile")
    student_id = st.number_input("Student ID", min_value=1, value=1001)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    ssc = st.slider("SSC Marks (%)", 0, 100, 70)
    hsc = st.slider("HSC Marks (%)", 0, 100, 68)

with col2:
    st.subheader("🧠 Skills & Training")
    internships = st.number_input("Internships", 0, 10, 1)
    projects = st.number_input("Projects", 0, 10, 2)
    certifications = st.number_input("Workshops / Certifications", 0, 10, 1)
    aptitude = st.slider("Aptitude Test Score", 0, 100, 60)
    softskills = st.slider("Soft Skills Rating", 0.0, 10.0, 6.5)

# Extra activities
st.subheader("🏆 Additional Factors")
col3, col4 = st.columns(2)

with col3:
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
with col4:
    placement_training = st.selectbox("Placement Training Attended", ["Yes", "No"])

# Encode Yes / No
extracurricular = 1 if extracurricular == "Yes" else 0
placement_training = 1 if placement_training == "Yes" else 0

# ---------------- PREDICTION ----------------
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🔍 Predict Placement Status", use_container_width=True):

    input_data = np.array([[  
        cgpa,
        internships,
        projects,
        certifications,
        aptitude,
        softskills,
        extracurricular,
        placement_training,
        ssc,
        hsc
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction == 1:
        st.success("✅ **Prediction: Student is LIKELY TO BE PLACED** 🎉")
    else:
        st.error("❌ **Prediction: Student is NOT LIKELY TO BE PLACED**")

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1] * 100
        st.info(f"📊 **Placement Probability:** {prob:.2f}%")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with ❤️ using Machine Learning & Streamlit  
    </p>
    """,
    unsafe_allow_html=True
)
