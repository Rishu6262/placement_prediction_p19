import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Placement Prediction AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#00C853;'>🎓 Placement Prediction System</h1>
<p style='text-align:center; font-size:18px;'>
AI-powered prediction using Machine Learning
</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------
st.info("""
**Models Used:**  
✔ Logistic Regression  
✔ Decision Tree Classifier  
✔ Random Forest Classifier  

Final trained model is used for prediction.
""")

# ---------------- INPUT SECTION ----------------
st.subheader("📌 Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    student_id = st.number_input("Student ID", min_value=1, value=101)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    ssc = st.slider("SSC Marks (%)", 0, 100, 70)
    hsc = st.slider("HSC Marks (%)", 0, 100, 65)

with col2:
    internships = st.number_input("Internships", 0, 10, 1)
    projects = st.number_input("Projects", 0, 10, 2)
    certifications = st.number_input("Workshops / Certifications", 0, 10, 1)
    aptitude = st.slider("Aptitude Test Score", 0, 100, 60)

with col3:
    softskills = st.slider("Soft Skills Rating", 0.0, 10.0, 6.5)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    placement_training = st.selectbox("Placement Training Attended", ["Yes", "No"])

# Encode Yes / No
extracurricular = 1 if extracurricular == "Yes" else 0
placement_training = 1 if placement_training == "Yes" else 0

# ---------------- PREDICTION ----------------
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🚀 Predict Placement Status", use_container_width=True):

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

    prediction = model.predict(input_data)[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction == 1:
        st.success("✅ **Prediction Result: STUDENT WILL BE PLACED** 🎉")
    else:
        st.error("❌ **Prediction Result: STUDENT WILL NOT BE PLACED**")

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1] * 100
        st.info(f"📊 **Placement Probability:** {prob:.2f}%")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px;'>
Made with ❤️ using Machine Learning & Streamlit  
</p>
""", unsafe_allow_html=True)
