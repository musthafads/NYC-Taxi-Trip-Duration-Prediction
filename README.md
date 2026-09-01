# 🚕 NYC Taxi Trip Duration Prediction

## 📌 Project Overview

This project predicts the **duration of a taxi trip in New York City** using Machine Learning.

The application allows users to enter taxi trip details such as pickup location, drop-off location, passenger count, and other trip-related information. The trained Machine Learning model then predicts the expected taxi trip duration.

The model is integrated with a **Streamlit web application** to provide a simple and user-friendly interface.

---

## 🎯 Project Objective

The main objective of this project is to:

- Predict NYC taxi trip duration.
- Apply Machine Learning to real-world transportation data.
- Perform data preprocessing and feature engineering.
- Train and evaluate a Machine Learning model.
- Deploy the model as an interactive Streamlit application.

---

## 📊 Dataset

The project is based on the **NYC Taxi Trip Duration** dataset.

Important features include:

- Vendor ID
- Pickup Date & Time
- Drop-off Date & Time
- Passenger Count
- Pickup Longitude
- Pickup Latitude
- Drop-off Longitude
- Drop-off Latitude
- Store and Forward Flag
- Trip Duration

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git & GitHub

---

## 🤖 Machine Learning

The project includes:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Model Saving
8. Streamlit Deployment

The trained model is saved using a `.pkl` file and loaded into the Streamlit application for prediction.

---

## 📁 Project Structure

```text
NYC-Taxi-Trip-Duration-Prediction/
│
├── app.py
├── requirements.txt
├── taxi_feature_columns.pkl
├── taxi_model_info.pkl
├── taxi_trip_duration_model.pkl
├── .gitignore
└── README.md
