import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Load model and preprocessor
model = joblib.load('../models/model.pkl')
preprocessor = joblib.load('../models/preprocessor.pkl')

# App title
st.title("Customer Churn Predictor")
st.write("Enter customer details to predict churn probability")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

# More inputs below columns
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Predict button
if st.button("Predict Churn"):
    
    # Create dataframe from inputs
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Add engineered features
    input_data['tenure_bucket'] = pd.cut(input_data['tenure'], 
                                          bins=[0, 12, 36, 72], 
                                          labels=['New', 'Established', 'Loyal'])
    
    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_data['total_services'] = input_data[service_cols].apply(
        lambda x: (x == 'Yes').sum(), axis=1)
    
    input_data['monthly_tenure_ratio'] = input_data['MonthlyCharges'] / (input_data['tenure'] + 1)
    input_data['is_fiber'] = (input_data['InternetService'] == 'Fiber optic').astype(int)
    input_data['has_security_support'] = ((input_data['OnlineSecurity'] == 'Yes') | 
                                           (input_data['TechSupport'] == 'Yes')).astype(int)
    input_data['is_high_risk'] = ((input_data['Contract'] == 'Month-to-month') & 
                                   (input_data['tenure'] < 12)).astype(int)
    
    # Preprocess and predict
    input_processed = preprocessor.transform(input_data)
    probability = model.predict_proba(input_processed)[0, 1]
    
    # Apply optimized threshold
    threshold = 0.20
    prediction = "Yes" if probability >= threshold else "No"
    
    # Display results
    st.subheader("Prediction Results")
    
    # Color-coded probability
    if probability >= 0.5:
        risk_level = "HIGH"
        color = "red"
        action = "Immediate retention offer recommended"
    elif probability >= 0.3:
        risk_level = "MEDIUM"
        color = "orange"
        action = "Proactive engagement recommended"
    else:
        risk_level = "LOW"
        color = "green"
        action = "Standard service monitoring"
    
    st.metric("Churn Probability", f"{probability:.1%}")
    st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
    st.info(f"**Recommended Action:** {action}")


# Sidebar with model info
st.sidebar.header("About")
st.sidebar.write("""
This app predicts customer churn using a Logistic Regression model 
trained on the Telco Customer Churn dataset.

**Model Performance:**
- AUC-ROC: 0.835
- Recall: 96%
- Optimized Threshold: 0.20

**Cost Optimization:**
- Reduces missed churners from 160 to 15
- Saves $49,700 in business costs
""")

st.sidebar.header("Risk Levels")
st.sidebar.write("""
- 🔴 **HIGH** (≥50%): Immediate action needed
- 🟠 **MEDIUM** (30-50%): Proactive engagement
- 🟢 **LOW** (<30%): Standard monitoring
""")