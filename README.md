# 🎓 Student Placement Prediction System

## 📌 Project Overview

The Student Placement Prediction System is a Machine Learning project that predicts whether a student is likely to be placed based on academic performance, internships, projects, certifications, aptitude test scores, soft skills, and placement training.

The project helps students understand the key factors that influence placement opportunities and enables data-driven career preparation.

The dataset contains information about **10,000 students** and their placement outcomes.

---

# 🚀 Objectives

* Predict student placement status using Machine Learning.
* Analyze factors affecting placement opportunities.
* Compare multiple classification algorithms.
* Identify the best-performing model.
* Provide insights for career improvement.

---

# 📊 Dataset Information

### Dataset Name

Placement Data Dataset

### Total Records

* 10,000 Students

### Total Features

* 12 Features

---

## Features Description

| Feature                   | Description                          |
| ------------------------- | ------------------------------------ |
| StudentID                 | Unique Student Identifier            |
| CGPA                      | College CGPA                         |
| Internships               | Number of Internships Completed      |
| Projects                  | Number of Academic/Personal Projects |
| Workshops/Certifications  | Number of Certifications             |
| AptitudeTestScore         | Aptitude Test Performance            |
| SoftSkillsRating          | Communication & Soft Skills Rating   |
| ExtracurricularActivities | Participation in Activities          |
| PlacementTraining         | Placement Training Attended          |
| SSC_Marks                 | Class 10 Marks                       |
| HSC_Marks                 | Class 12 Marks                       |
| PlacementStatus           | Target Variable (Placed/Not Placed)  |

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Streamlit

---

# 📂 Project Structure

```bash
Placement_Prediction_Project/
│
├── app.py
├── model.pkl
├── placementdata.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── placement_prediction.ipynb
│
└── assets/
    └── screenshots/
```

---

# 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

### Academic Analysis

* CGPA Distribution
* SSC Marks Analysis
* HSC Marks Analysis

### Career Analysis

* Internship Impact
* Projects Impact
* Certifications Analysis

### Skills Analysis

* Aptitude Test Score Analysis
* Soft Skills Rating Analysis

### Placement Analysis

* Placed vs Not Placed Students
* Placement Training Impact
* Extracurricular Activities Impact

---

# 🤖 Machine Learning Models Used

Multiple machine learning algorithms were trained and compared.

## 1. Logistic Regression

Advantages:

* Fast Training
* Easy Interpretation
* Strong Baseline Model

---

## 2. Decision Tree

Advantages:

* Handles Nonlinear Data
* Easy Visualization
* Feature Importance Analysis

---

## 3. Random Forest

Advantages:

* High Accuracy
* Reduces Overfitting
* Robust Predictions

---

## 4. K-Nearest Neighbors (KNN)

Advantages:

* Simple and Effective
* Classification Based on Similar Data Points
* Works Well with Scaled Data

---

# ⚙️ Data Preprocessing

### Missing Value Handling

* Checked for Null Values
* Removed Inconsistent Records

### Encoding

Categorical Variables Converted Using:

* Label Encoding

### Feature Scaling

Applied:

```python
StandardScaler()
```

for numerical features.

---

# 📈 Model Evaluation Metrics

The following metrics were used:

### Accuracy

```python
accuracy_score()
```

### Precision

```python
precision_score()
```

### Recall

```python
recall_score()
```

### F1 Score

```python
f1_score()
```

### Confusion Matrix

```python
confusion_matrix()
```

---

# 🏆 Best Model Selection

Models were compared based on:

* Accuracy
* Precision
* Recall
* F1 Score
* Generalization Performance

Models Compared:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)

The model with the best evaluation performance was selected for deployment.

---

# 📊 Important Placement Factors

The most influential factors include:

1. CGPA
2. Aptitude Test Score
3. Soft Skills Rating
4. Internships
5. Projects
6. Workshops & Certifications
7. Placement Training
8. SSC Marks
9. HSC Marks
10. Extracurricular Activities

---

# 💻 Streamlit Web Application

The project includes a Streamlit-based web application where users can:

### Input Student Information

* CGPA
* Internships
* Projects
* Certifications
* Aptitude Score
* Soft Skills Rating
* Placement Training
* Academic Marks

### Get Prediction

The system predicts:

* Placement Status
* Placement Probability

in real time.

---


### Run Application

```bash
streamlit run app.py
```

# 📦 Requirements

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

# 🎯 Learning Outcomes

Through this project, I learned:

* Data Cleaning
* Data Visualization
* Feature Engineering
* Classification Algorithms
* Model Evaluation
* Hyperparameter Tuning
* Streamlit Deployment
* End-to-End Machine Learning Workflow

---

# 🔮 Future Improvements

* Deep Learning Models
* Resume Analysis
* Interview Readiness Score
* Career Recommendation System
* Placement Probability Dashboard
* Student Performance Analytics

---

# 📜 Disclaimer

This project is developed for educational and research purposes only.

The placement predictions generated by this system are based on machine learning models trained on historical student data. The results are intended to provide insights and demonstrate the practical application of data science and machine learning techniques.

Actual placement outcomes may vary depending on interview performance, technical skills, communication abilities, company requirements, and market conditions.

---

# 👨‍💻 Author

**Rishu Gurjar**

Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

### Skills

* Python
* SQL
* Machine Learning
* Data Analysis
* Streamlit
* Scikit-Learn

Connect with me on LinkedIn and GitHub to explore more Data Science and Machine Learning projects.
