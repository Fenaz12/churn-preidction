import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.title(" Customer Churn Prediction Dashboard")
st.write("Enter the customer's details below to predict their likelihood of churning.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.selectbox("Country", ['Bangladesh', 'Canada', 'Germany', 'Australia', 'India', 'USA', 'UK'])
    
    customer_segment = st.selectbox("Customer Segment", ["Regular", "Premium", "Enterprise"])

with col2:
    st.subheader("Subscription & Usage")
    tenure_months = st.slider("Tenure (Months)", 0, 120, 12)
    contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Annual"])
    monthly_fee = st.number_input("Monthly Fee ($)", value=50.0)
    total_revenue = st.number_input("Total Revenue ($)", value=600.0)
    monthly_logins = st.slider("Monthly Logins", 0, 100, 15)
    weekly_active_days = st.slider("Weekly Active Days", 0, 7, 3)
    avg_session_time = st.number_input("Avg Session Time (mins)", value=25.0)

with col3:
    st.subheader("Support & Satisfaction")
    support_tickets = st.slider("Support Tickets", 0, 20, 1)
    escalations = st.slider("Escalations", 0, 10, 0)
    avg_resolution_time = st.number_input("Avg Resolution Time (hours)", value=12.0)
    csat_score = st.slider("CSAT Score (1-5)", 1.0, 5.0, 4.0, 0.5)
    payment_method = st.selectbox("Payment Method", ["Credit_Card", "PayPal", "Bank_Transfer"])
    complaint_type = st.selectbox("Complaint Type", ["No_Complaint", "Billing", "Technical", "Service"])

if st.button("Predict Churn Risk", type="primary"):
    customer_data = {
        "age": age,
        "gender": gender,
        "country": country,
        "customer_segment": customer_segment,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_fee": monthly_fee,
        "total_revenue": total_revenue,
        "monthly_logins": monthly_logins,
        "weekly_active_days": weekly_active_days,
        "avg_session_time": avg_session_time,
        "support_tickets": support_tickets,
        "escalations": escalations,
        "avg_resolution_time": avg_resolution_time,
        "csat_score": csat_score,
        "payment_method": payment_method,
        "complaint_type": complaint_type
    }
    
    try:
        response = requests.post(API_URL, json=customer_data)
        
        if response.status_code == 200:
            result = response.json()
            
            st.divider()
            
            if result["risk_level"] == "High":
                st.error(f"🚨 **HIGH RISK OF CHURN**")
            else:
                st.success(f"✅ **LOW RISK OF CHURN**")
                
            st.metric(label="Churn Probability", value=f"{result['churn_probability'] * 100:.1f}%")
            
        else:
            st.warning(f"Error from API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure your FastAPI server is running!")