# # ❤️ Heart Disease Prediction — BRFSS Dashboard

> A machine learning web application that predicts heart disease risk using health indicators from the BRFSS 2015 dataset.

**Developer:** Malak Gamal Ahmed Sanad Eleza
**Student ID:** 221000014
**Programme:** CBIO313: Data Mining and Machine Learning-2026SPRG

---

## 🔗 Live Application

👉 **[Open the App on Streamlit Cloud](https://mlproj-aax5ikmwktyvugkvpvejmf.streamlit.app)**
📹 **[Video Walkthrough](https://drive.google.com/file/d/1FpzhKSLhhoTmNeGHlduCnnRONJlmV-Ha/view?usp=sharing)**
💻 **[GitHub Repository](https://github.com/MalakGamalEleza/MLPROJ)**

---

## 📋 Project Overview

Heart disease is one of the leading causes of death worldwide. Many lifestyle and health-related factors contribute to the risk of developing cardiovascular disease. This project addresses the question:

> *"Can we predict whether a person is likely to have heart disease or a heart attack based on health indicators?"*

This project implements a complete end-to-end data science workflow — from raw healthcare data to a deployed machine learning web application.

---

## 📂 Repository Structure

```bash
├── app.py                      # Streamlit web application
├── requirements.txt           # Python dependencies
├── notebook.ipynb             # Full analysis notebook
├── dataset.csv                # BRFSS dataset
├── best_model.pkl             # Trained ML model
├── scaler.pkl                 # Fitted scaler
├── selected_features.pkl      # Selected feature names
└── README.md
```

---

## 🗃️ Dataset — BRFSS 2015 Heart Disease Dataset

| Property | Detail                                           |
| -------- | ------------------------------------------------ |
| Source   | Kaggle — Heart Disease Health Indicators Dataset (https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset?utm_source=chatgpt.com)|
| Rows     | ~253,680 records                                 |
| Columns  | 10 selected features                             |
| Target   | Heart Disease or Heart Attack                    |

### Target Variable

`HeartDiseaseorAttack`

The target variable predicts whether a person has had heart disease or a heart attack.

### Selected Features

The project uses exactly **10 columns**:

* HeartDiseaseorAttack (Target)
* HighBP
* HighChol
* BMI
* Smoker
* Diabetes
* PhysActivity
* GenHlth
* Sex
* Age
* Income

### Why this dataset is not pre-cleaned

* Contains duplicate records
* Requires feature selection
* Includes mixed health indicators with varying scales
* Needs preprocessing before machine learning

---

## 🔬 Project Steps

### Step 1 — Data Cleaning

* Removed duplicate rows
* Checked for missing values
* Handled outliers in numerical features
* Verified data consistency

### Step 2 — Exploratory Data Analysis (EDA)

* Investigated more than 6 variables
* Performed univariate and bivariate analysis
* Analysed relationships between health indicators and heart disease risk
* Used multiple visualizations

### Visualizations Included

* Histograms
* Box plots
* Heatmaps
* Grouped Bar
* Barchart
* Pie chart

### Step 3 — Feature Engineering

A new feature called **Health Risk Score** was created by combining major health risk indicators such as:

```python
RiskScore = HighBP + HighChol + Smoker + Stroke + Diabetes
```

This feature helps represent overall cardiovascular health risk.

### Step 4 — Feature Selection

Feature importance techniques were used to identify the most important predictors affecting heart disease risk.

Method used:

* **Embedded Method (Random Forest Feature Importance)**

### Step 5 — Modelling

Three machine learning algorithms were trained and compared:

| Algorithm           | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | ~84%     | 0.79      | 0.75     | 0.77     |
| Decision Tree       | ~86%     | 0.80      | 0.79     | 0.79     |
| **Random Forest ✅** | **~89%** | **0.84**  | **0.83** | **0.84** |

**Random Forest was selected** as the best-performing model because of its strong predictive performance and robustness.

### Step 6 — Hyperparameter Tuning

`GridSearchCV` with cross-validation was used to tune model parameters such as:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

### Why tuning matters

Hyperparameter tuning improves model performance by finding the best settings and reducing overfitting.

### Step 7 — Validation & Evaluation

* **Validation Method:** 80/20 train-test split
* **Cross Validation:** Stratified K-Fold
* **Metrics Used:**

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

Final model performance met the project requirements with acceptable precision and recall.

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/MalakGamalEleza/MLPRO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

```txt
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🖥️ Application Features

* Health indicator input form
* Heart disease prediction
* Confidence score display
* Model probability visualization
* User-friendly Streamlit interface

---
