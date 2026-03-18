# 🌍 AI Epidemic Forecasting Dashboard

## 📌 Project Overview

This project aims to predict the spread of infectious diseases (such as COVID-19) using historical epidemiological data and machine learning models. It provides an interactive dashboard to visualize future case predictions, identify hotspot regions, and analyze factors such as vaccination and mobility.

---

## 🎯 Objectives

* Predict future COVID-19 cases (next 7–14 days)
* Identify high-risk (hotspot) regions
* Analyze impact of vaccination and mobility
* Provide interactive visualizations through a dashboard

---

## 🧠 Features

* 📈 Time-series forecasting (LSTM / ARIMA / XGBoost)
* 🔥 Hotspot detection (High / Medium / Low risk)
* 🌍 Interactive world map visualization
* 💉 Vaccination impact analysis
* 🚶 Mobility-based spread analysis
* 📊 Streamlit dashboard for real-time interaction

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow
* **Visualization:** Plotly, Matplotlib
* **Dashboard:** Streamlit
* **Version Control:** Git

---

## 📂 Project Structure

```
epidemic_prediction_project
│
├── data_raw/              # Raw datasets
├── data_processed/        # Cleaned & processed datasets
├── notebooks/             # Jupyter notebooks (data preprocessing, EDA)
├── models/                # ML/DL models
├── visualization/         # Graphs & plots
├── dashboard/             # Streamlit app
└── README.md
```

---

## 📊 Datasets Used

* Johns Hopkins COVID-19 Time Series Dataset
* Our World in Data (OWID) COVID Dataset
* Google Mobility Reports (optional)

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone <your-repo-link>
cd epidemic_prediction_project
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run preprocessing:

```
python data_pipeline.py
```

### Run dashboard:

```
streamlit run app.py
```

---

## 📈 Output

* Predicted case trends for selected countries
* Risk classification (High / Medium / Low)
* Interactive global heatmap

---

## 👥 Team Members

* Himanshu Sharma – Data Engineering & Preprocessing
* Samarthya – Model Development
* [Add others]

---

## 🚀 Future Enhancements

* Real-time data integration
* Multi-disease prediction
* Advanced deep learning models
* Deployment on cloud (AWS/Heroku)

---

## 📜 License

This project is developed for academic and hackathon purposes.
