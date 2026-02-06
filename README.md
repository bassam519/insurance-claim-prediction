# 📊 Insurance Claims Prediction — Machine Learning Project 🚗🤖

## 🧠 Project Overview

This project presents a complete end-to-end Machine Learning pipeline to predict whether a customer will file an **insurance claim** based on vehicle, customer, and policy features. It demonstrates a full Business Intelligence & AI workflow including data cleaning, feature engineering, preprocessing pipelines, class imbalance handling, and ensemble modeling.

The system was designed to be **production-ready**, with reusable preprocessing pipelines and saved model artifacts.

---

## 🎯 Problem Statement

Insurance companies need accurate claim prediction to:

* Reduce financial risk 💰
* Improve pricing strategies 📈
* Identify high-risk customers early ⚠️
* Support data-driven underwriting decisions 🏦

**Objective:**
Build a classification model that predicts:

> `claim_status` → whether a policy holder will file a claim or not.

---

## 🗂️ Dataset Description

The dataset contains structured insurance and vehicle information, including:

* 👤 Customer age
* 🚗 Vehicle age
* 🌍 Region & population density
* 🔧 Engine and power specifications
* ⛽ Fuel type
* 🛡️ Safety features (ESC, airbags, TPMS, etc.)
* ⚙️ Transmission & steering type
* 📦 Vehicle dimensions and weight
* 🧾 Policy subscription length

**Target Variable:** Claim Status (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand data quality and distributions:

* ✅ Missing value analysis
* ✅ Duplicate record checks
* ✅ Statistical summaries
* ✅ Class distribution plots
* ✅ Count plots and pie charts
* ✅ Outlier detection using IQR
* ✅ Boxplots after outlier treatment

Outliers were handled using the **IQR clipping method** to improve model stability.

---

## 🛠️ Feature Engineering

Text-based mechanical specifications were transformed into numerical features:

Extracted from:

* `max_torque`
* `max_power`

Created new features:

* torque_nm
* torque_rpm
* power_nm
* power_rpm

This conversion improved model learning by turning text specs into numeric signals 🔬

---

## ⚙️ Data Preprocessing Pipeline

A full preprocessing pipeline was built using **ColumnTransformer + Scikit-Learn Pipelines**.

### 🔢 Numerical Features

* Median imputation

### 🟢 Binary Categorical Features

* Most frequent imputation
* Ordinal encoding

### 🟡 Nominal Categorical Features

* One-Hot Encoding

### 🎯 High-Cardinality Features

* Target Encoding for better signal extraction

Pipeline saved for reuse:

```
joblib.dump(preprocessor, "insurance_preprocessor.pkl")
```

---

## ⚖️ Handling Class Imbalance

The dataset showed class imbalance in claim labels.

Applied:

* **SMOTE (Synthetic Minority Over-sampling Technique)**

This improves minority class prediction performance and recall 🎯

---

## 🤖 Models Used

An ensemble model using **Soft Voting** was built from three strong learners:

### 🌲 Random Forest

* Robust and noise-resistant
* Multiple decision trees

### 🚀 XGBoost

* Gradient boosting algorithm
* Excellent performance on structured data

### 🌳 HistGradientBoosting

* Fast boosting method
* Efficient with large datasets

---

## 🧩 Ensemble Strategy

Used **Weighted Soft Voting**:

```
weights = [2, 2, 3]
```

HistGradientBoosting received slightly higher weight for stronger influence 🏆

This improves stability and generalization.

---

## 🧪 Training Setup

* Train/Test Split: 80% / 20%
* Random State: 42
* Full pipeline included:

  * Preprocessing
  * SMOTE balancing
  * Ensemble classifier

Unified pipeline structure:

```
Preprocessor → SMOTE → Voting Classifier
```

---

## 📏 Evaluation Metrics

Models were evaluated using:

* ✅ Accuracy
* ✅ F1 Score
* ✅ ROC-AUC
* ✅ Confusion Matrix

### 📊 Final Model Performance

Fill with your notebook results:

* Accuracy: **XX%**
* F1 Score: **XX**
* ROC-AUC: **XX**

Confusion matrix visualization used for detailed error analysis 🔍

---

## 📚 Libraries & Tools Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* xgboost
* category_encoders
* joblib

---

## 🏗️ Project Workflow

```
Raw Data
 ↓
EDA & Cleaning
 ↓
Feature Engineering
 ↓
Preprocessing Pipeline
 ↓
SMOTE Balancing
 ↓
Ensemble Models
 ↓
Evaluation
 ↓
Saved Model & Pipeline
```

---

## 💾 Saved Artifacts

* ✅ Preprocessing pipeline file
* ✅ Trained ensemble model
* ✅ Engineered dataset version

Ready for deployment and reuse 🚀

---

## 🧠 Key Learning Outcomes

* Building advanced preprocessing pipelines 🧩
* Handling imbalanced datasets ⚖️
* Feature engineering from text attributes 🔬
* Ensemble learning strategies 🤝
* Weighted soft voting models 🏆
* Production-style ML workflow 🏗️

---

**Author:** CS & AI Student — Business Intelligence Project
