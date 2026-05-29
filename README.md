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
| Layer | Tools |
|-------|-------|
| ML | XGBoost, Scikit-learn, SMOTE |
| Data | Pandas, NumPy |
| Visualization | Plotly, Streamlit |
| Deployment | Streamlit Cloud |

## 📂 Project Structure
\`\`\`
customer-churn-prediction/
├── app.py                    # Streamlit dashboard
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── models/                   # Saved artifacts (gitignored)
├── data/                     # Dataset (gitignored)
├── requirements.txt
└── README.md
\`\`\`

## 🏃 Run Locally
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction
cd customer-churn-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv', 'data/telco_churn.csv')"

# Train model
python -m src.model_training

# Launch dashboard
streamlit run app.py
\`\`\`

## 📈 Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.84 |
| F1 Score | ~0.62 |
| Precision | ~0.67 |
| Recall | ~0.57 |

## 🔍 Key Insights
- Month-to-month contracts churn at 3× the rate of annual contracts
- Fiber optic users show highest churn despite premium pricing
- Customers without online security churn 35% more
- First 12 months is the critical retention window

## 📄 License
MIT