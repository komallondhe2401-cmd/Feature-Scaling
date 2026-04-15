# 📊 Feature Scaling Techniques in Data Preprocessing

## 📌 Project Overview

This project focuses on **Feature Scaling**, an essential step in data preprocessing for Machine Learning. Different scaling techniques were applied to transform the dataset so that all features contribute equally to the model performance.

---

## 🎯 Objectives

* To understand the importance of feature scaling
* To apply different scaling techniques on the dataset
* To compare various normalization and standardization methods

---

## 🛠️ Tools & Libraries Used

* Python 🐍
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Dataset

* Dataset Name: `train.csv`
* Source: Kaggle
* Description: Dataset used for preprocessing and scaling of numerical features

---

## ⚙️ What is Feature Scaling?

Feature Scaling is a technique used to **standardize the range of independent variables** so that no feature dominates due to its scale.

---

## 📊 Techniques Used

### 🔹 1. Standardization

Standardization transforms data to have:

* Mean = 0
* Standard Deviation = 1

#### ✅ Method Used:

* **StandardScaler**

```python id="std123"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

---

### 🔹 2. Normalization

Normalization scales data to a specific range (usually 0 to 1).

#### ✅ Methods Used:

### a) Min-Max Scaling

* Scales data between 0 and 1

```python id="mm123"
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df)
```

---

### b) Mean Normalization

* Centers data around mean

```python id="mn123"
df_mean_norm = (df - df.mean()) / (df.max() - df.min())
```

---

### c) MaxAbs Scaling

* Scales data between -1 and 1
* Works well with sparse data

```python id="ma123"
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df_maxabs = scaler.fit_transform(df)
```

---

### d) Robust Scaling

* Uses median and IQR (less affected by outliers)

```python id="rb123"
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_robust = scaler.fit_transform(df)
```

---

## 🔍 Key Insights

* Standardization is useful when data follows normal distribution
* Min-Max scaling is best when data has fixed bounds
* Robust scaling performs well in presence of outliers
* Different scaling methods impact model performance differently

---

## 📈 Conclusion

Feature scaling is a crucial preprocessing step that improves the performance and accuracy of machine learning models. Different scaling techniques were successfully applied and compared.

---

## 🚀 Future Work

* Apply scaling before training ML models
* Compare model performance with different scaling techniques
* Automate preprocessing pipeline
* 
---
