# 📊 Customer Churn Prediction & Intelligence Dashboard

An end-to-end ML project predicting telecom customer churn using XGBoost, with a full business dashboard built in Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo
[👉 View Dashboard](YOUR_STREAMLIT_URL_HERE)

## 🎯 Business Problem
Telecom companies lose 15-25% of customers annually. Acquiring a new customer costs 5-7× more than retaining one. This dashboard identifies at-risk customers before they churn and quantifies the revenue impact.

## 💡 Key Features
- **Churn Prediction** — XGBoost model with ROC-AUC ~0.84
- **Segment Analysis** — Churn by contract type, payment method, services
- **Risk Scoring** — Low / Medium / High risk tiering per customer
- **Live Predictor** — Enter any customer's details, get instant churn probability
- **Business KPIs** — Revenue at risk, CLV impact, retention levers

## 🛠️ Tech Stack

| Layer         | Tools                          |
|---------------|--------------------------------|
| ML            | XGBoost, Scikit-learn, SMOTE   |
| Data          | Pandas, NumPy                  |
| Visualization | Plotly, Streamlit              |
| Deployment    | Streamlit Cloud                |

## 📂 Project Structure
customer-churn-prediction/
│
├── app.py                      # Streamlit dashboard
│
├── src/
│   ├── init.py
│   ├── data_preprocessing.py   # Cleaning, encoding, feature engineering
│   ├── model_training.py       # XGBoost training + evaluation
│   └── utils.py                # Shared helpers
│
├── data/                       # ← NOT in repo (see Dataset Setup below)
│   └── telco_churn.csv
│
├── models/                     # ← NOT in repo (auto-generated after training)
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── test_predictions.csv
│
├── requirements.txt
├── setup.py
├── .env
├── .env.example
├── .gitignore
└── README.md

## 📥 Dataset Setup (Required Before Running)

The dataset is **not included in this repo** (gitignored to keep the repo lightweight).
Download it manually from Kaggle:

1. Go to 👉 [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Click **Download** (free Kaggle account required)
3. Unzip the downloaded file
4. Place the CSV inside the `data/` folder
5. Rename it to `telco_churn.csv`

Your folder should look like:

customer-churn-prediction/
└── data/
└── telco_churn.csv    ✅

> The `data/` folder is listed in `.gitignore` and will never be accidentally pushed to GitHub.

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction
cd customer-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env

# 5. Download dataset from Kaggle (see Dataset Setup above)

# 6. Train the model
python -m src.model_training

# 7. Launch the dashboard
streamlit run app.py
```

## 📈 Model Performance

| Metric    | Score |
|-----------|-------|
| ROC-AUC   | ~0.84 |
| F1 Score  | ~0.62 |
| Precision | ~0.67 |
| Recall    | ~0.57 |

## 🔍 Key Insights
- Month-to-month contracts churn at **3× the rate** of annual contracts
- Fiber optic users show highest churn despite premium pricing
- Customers without online security churn **35% more**
- First **12 months** is the critical retention window

## 🌐 Deployment (Streamlit Cloud)

This app auto-trains the model on first deploy via `setup.py`.
No manual steps needed — just connect your GitHub repo and click Deploy.

## 📄 License
MIT