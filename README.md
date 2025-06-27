# 📊 Customer Churn Prediction App

This is an end-to-end machine learning project that predicts whether a telecom customer will churn (leave the service) based on their service usage and demographic data. Built using **Random Forest** and deployed as a user-friendly **Streamlit web app**, this tool demonstrates real-world churn modeling and interactive ML visualization.

---

## 🚀 Features

- End-to-end ML pipeline using the **Telco Customer Churn dataset**
- Exploratory Data Analysis (EDA) using Seaborn and Matplotlib
- Label encoding of categorical variables and handling of missing values
- Trained a **Random Forest Classifier** for binary classification
- Achieved high accuracy with ROC-AUC evaluation
- Saved model and label encoders using `pickle` for deployment
- Built and deployed an interactive **Streamlit web app**
- Predicts churn in real-time from user-provided inputs

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn (Random Forest, LabelEncoder)
- Matplotlib & Seaborn (EDA)
- Streamlit (Frontend UI)
- Pickle (model and encoder storage)

---

## 📂 Project Structure
customer-churn-prediction/
├── app.py # Streamlit app code
├── customer_churn_rfc.pkl # Trained Random Forest model with features
├── encoders.pkl # Dictionary of LabelEncoders used during training
├── requirements.txt # Dependencies for running the app
└── README.md # This file


---

## 📈 Model Overview

- **Dataset**: IBM Telco Customer Churn Dataset
- **Target Variable**: `Churn` (Yes/No)
- **Model Used**: Random Forest Classifier
- **Performance Metrics**: Accuracy, Confusion Matrix, ROC-AUC
- **Preprocessing**:
  - Converted `TotalCharges` to numeric
  - Encoded categorical features using `LabelEncoder`
  - Handled missing values
  - Saved label encoders for consistent deployment preprocessing

---

## 🌐 Live Demo

🔗 _[Add your Streamlit URL here after deployment]_

---

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/customer-churn-streamlit.git
   cd customer-churn-streamlit
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the app
   ```bash
   streamlit run app.py
