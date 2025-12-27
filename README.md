# Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company, optimized for business cost reduction.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)

## 🎯 Business Impact

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Churners Caught | 57% | 96% | +39% |
| Missed Churners | 160 | 15 | -145 |
| Business Cost | $85,750 | $36,050 | -$49,700 |

## 📋 Problem Statement

Telecom company loses customers (churn). Each lost customer costs ~$500 in revenue. False alarms (offering retention to non-churners) cost ~$50.

**Goal:** Build a model that catches churners while minimizing total business cost.

## 🔍 Approach

1. **EDA**: Identified key churn drivers (contract type, tenure, fiber optic service)
2. **Feature Engineering**: Created 6 new features (tenure buckets, service count, high-risk flag)
3. **Baseline Models**: Logistic Regression (AUC: 0.84) outperformed Random Forest (AUC: 0.82)
4. **Cost Optimization**: Used class weights + threshold tuning to minimize business cost

## 📊 Results

### Model Performance

| Model | AUC-ROC | Recall | Precision |
|-------|---------|--------|-----------|
| Baseline (LR) | 0.836 | 57.2% | 65.1% |
| Optimized (LR + Class Weights) | 0.835 | 96.0% | 38.6% |

### Key Insight

Accuracy dropped from 80% to 58%, but **business cost dropped by $49,700**. 

Optimizing for the right metric matters more than chasing accuracy.

### Top Churn Predictors

1. **Contract Type**: Month-to-month customers churn 4x more than two-year contracts
2. **Tenure**: New customers (<12 months) have highest churn risk
3. **Fiber Optic**: Surprisingly high churn (42%) vs DSL (14%)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### Run the App
```bash
cd app
streamlit run app.py
```

### Run Notebooks

Notebooks are in `notebooks/` folder:
1. `01_eda.ipynb` - Exploratory Data Analysis
2. `02_preprocessing.ipynb` - Data preprocessing
3. `03_baseline_models.ipynb` - Model training and evaluation
4. `04_feature_engineering.ipynb` - Feature creation
5. `05_class_imbalance.ipynb` - Cost-sensitive optimization

## 📁 Project Structure
```
customer-churn-prediction/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   └── README.md              # Data source information
├── models/
│   ├── model.pkl              # Trained model
│   └── preprocessor.pkl       # Fitted preprocessor
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_class_imbalance.ipynb
├── README.md
└── requirements.txt
```

## 💡 Key Learnings

1. **Feature engineering doesn't always help** - Baseline model already captured patterns; engineered features added minimal value

2. **Class imbalance is the real problem** - With 73/27 split, model defaulted to predicting "no churn"

3. **Optimize for business metrics** - Accuracy is misleading; cost-based optimization saved $49,700

4. **Simple models can win** - Logistic Regression beat Random Forest due to linear relationships in data

## 📊 Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle (7,043 customers, 21 features)

## 👤 Author

Your Name - [LinkedIn](https://www.linkedin.com/in/ananthakrishnan-j/) - [GitHub](https://github.com/Ananthakrishnan-Jayadev) 