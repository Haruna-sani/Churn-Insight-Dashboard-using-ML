# 📎 ChurnSight: Customer Churn Prediction Dashboard

## 📌 Overview

Customer retention is a core concern for subscription-based businesses. Acquiring new customers is expensive—making it essential to understand **why customers leave (churn)** and proactively **identify those at risk**.

**ChurnSight** is an interactive machine learning dashboard that empowers business stakeholders to:

* Predict customer churn using an Artificial Neural Network (ANN)
* Analyze customer behavior by segment
* Identify high-risk groups
* Support strategic retention efforts

---

## 💼 Business Problem

Many businesses face **high customer churn rates**, which significantly impact profitability. With customer lifetime value diminishing, it becomes crucial to:

* Identify customers likely to churn
* Understand the drivers behind their behavior
* Take early action to reduce churn through targeted campaigns

### 📈 Objectives:

* Build a predictive model to classify whether a customer will churn.
* Enable business stakeholders to visualize and explore churn risk interactively.
* Generate actionable insights based on customer characteristics and usage metrics.

---

## 🧪 Dataset & Key Insights

The dataset includes both numerical and categorical customer information:

* **Numerical**: Age, Tenure, Support Calls, Payment Delay, Total Spend, etc.
* **Categorical**: Gender, Subscription Type, Contract Length

### 🔎 Exploratory Data Analysis (EDA):

| Feature             | Churn Rate (%) |
| ------------------- | -------------- |
| Gender (Female)     | 67.0           |
| Subscription: Basic | 58.0           |
| Contract: Monthly   | 100.0          |

**High churn** is associated with:

* Frequent support calls
* Delayed payments
* Lower spending
* Shorter tenure

### 🔗 Feature Correlations with Churn:

* **Support Calls** → +0.57
* **Payment Delay** → +0.31
* **Total Spend** → −0.43

---

## 🛠️ Tech Stack and Libraries

### 🧮 Data Manipulation & Analysis

```python
import numpy as np
import pandas as pd
```

### 📊 Data Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

### ⚙️ Data Preprocessing & Evaluation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
```

### 🤖 Deep Learning

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

### 🚫 Warnings Management

```python
import warnings
warnings.filterwarnings("ignore")
```

---

## 🧠 Model Development

An **Artificial Neural Network (ANN)** was trained on a structured customer dataset with the following architecture:

* Input: Scaled numerical and encoded categorical features
* Hidden layers: Dense + Dropout for regularization
* Output: Sigmoid (binary churn prediction)

> Final Model Accuracy: **\~100%**
> Confusion Matrix: 28625 (No Churn), 37500 (Churn)

---

## 📊 ChurnSight Dashboard

Built with **Streamlit**, the dashboard allows stakeholders to:

### 🔍 Upload New Data

* Upload CSVs with customer details
* Automatically encode & normalize features

### 🤖 Predict Churn

* Display churn probabilities
* Download predictions as CSV

### 📈 Visualize Key Insights

* Pie/Bar charts by **gender**, **contract**, **subscription**
* Boxplots: **Support Calls**, **Spend**
* Violin plot: **Payment Delay**
* Correlation heatmaps
* High-churn segment analysis

### 🧠 Advanced Analytics

* Identify **segments with highest churn**
* Compare **spending patterns** across churn risk levels

---

## 📥 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/churnsight-dashboard.git
   cd churnsight-dashboard
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

4. Upload your CSV file and explore the results!

---

## 👥 Stakeholder Value

ChurnSight enables:

* **Marketing teams** to target at-risk customers
* **Customer success teams** to optimize support
* **Executives** to monitor retention KPIs in real-time

---

## 📎 Sample Outputs

* 📁 `churn_predictions.csv`: downloadable output of predicted churn
* 📊 Charts: insightful visuals categorized by gender, subscription, and more

---

## 📌 Future Enhancements

* Integrate SHAP for feature importance
* Time-series churn trend analysis
* Real-time API deployment

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open issues for feature requests or bugs.

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).
