import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.write("Enter the customer's details below to predict their likelihood of churning.")

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age               = st.slider("Age", 18, 100, 30)
    gender            = st.selectbox("Gender", ["Male", "Female"])
    country           = st.selectbox("Country", ["Bangladesh", "Canada", "Germany", "Australia", "India", "USA", "UK"])
    customer_segment  = st.selectbox("Customer Segment", ["Enterprise", "SME", "Individual"])
    signup_channel    = st.selectbox("Signup Channel", ["Web", "Mobile", "Referral"])
    discount_applied  = st.selectbox("Discount Applied?", ["No", "Yes"])
    price_increase_last_3m = st.selectbox("Price Increase Last 3 Months?", ["No", "Yes"])

with col2:
    st.subheader("Subscription & Usage")
    tenure_months        = st.slider("Tenure (Months)", 0, 120, 12)
    contract_type        = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"])
    monthly_fee          = st.number_input("Monthly Fee ($)", value=50.0)
    total_revenue        = st.number_input("Total Revenue ($)", value=600.0)
    monthly_logins       = st.slider("Monthly Logins", 0, 100, 15)
    weekly_active_days   = st.slider("Weekly Active Days", 0, 7, 3)
    avg_session_time     = st.number_input("Avg Session Time (mins)", value=25.0)
    features_used        = st.slider("Features Used", 1, 15, 4)
    usage_growth_rate    = st.slider("Usage Growth Rate", -0.6, 0.6, 0.0, step=0.01)
    last_login_days_ago  = st.slider("Days Since Last Login", 0, 80, 5)

with col3:
    st.subheader("Support & Satisfaction")
    support_tickets      = st.slider("Support Tickets", 0, 20, 1)
    escalations          = st.slider("Escalations", 0, 10, 0)
    avg_resolution_time  = st.number_input("Avg Resolution Time (hrs)", value=12.0)
    csat_score           = st.slider("CSAT Score (1-5)", 1.0, 5.0, 4.0, 0.5)
    payment_method       = st.selectbox("Payment Method", ["Card", "PayPal", "Bank_Transfer"])
    payment_failures     = st.slider("Payment Failures", 0, 5, 0)
    complaint_type       = st.selectbox("Complaint Type", ["No_Complaint", "Billing", "Technical", "Service"])
    nps_score            = st.slider("NPS Score", -100, 100, 20)
    email_open_rate      = st.slider("Email Open Rate", 0.1, 0.9, 0.5, 0.01)
    marketing_click_rate = st.slider("Marketing Click Rate", 0.01, 0.5, 0.25, 0.01)
    referral_count       = st.slider("Referral Count", 0, 7, 0)
    survey_response      = st.selectbox("Survey Response", ["Satisfied", "Neutral", "Unsatisfied"])

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", type="primary"):
    customer_data = {
        "age": age, "gender": gender, "country": country,
        "customer_segment": customer_segment, "tenure_months": tenure_months,
        "contract_type": contract_type, "monthly_fee": monthly_fee,
        "total_revenue": total_revenue, "monthly_logins": monthly_logins,
        "weekly_active_days": weekly_active_days, "avg_session_time": avg_session_time,
        "support_tickets": support_tickets, "escalations": escalations,
        "avg_resolution_time": avg_resolution_time, "csat_score": csat_score,
        "payment_method": payment_method, "complaint_type": complaint_type,
        "payment_failures": payment_failures, "last_login_days_ago": last_login_days_ago,
        "usage_growth_rate": usage_growth_rate, "nps_score": nps_score,
        "email_open_rate": email_open_rate, "marketing_click_rate": marketing_click_rate,
        "features_used": features_used, "referral_count": referral_count,
        "signup_channel": signup_channel, "discount_applied": discount_applied,
        "price_increase_last_3m": price_increase_last_3m, "survey_response": survey_response
    }

    try:
        response = requests.post(API_URL, json=customer_data)

        if response.status_code == 200:
            result = response.json()
            st.divider()

            # ── Risk result ───────────────────────────────────────────────────
            prob_pct = f"{result['churn_probability'] * 100:.1f}%"
            if result["risk_level"] == "High":
                st.error(f"🔴 HIGH RISK OF CHURN — Probability: {prob_pct}")
                st.caption("⚠️ *Note: The baseline churn rate for this business is ~10%. A probability above 15% indicates severe relative risk.*")
            else:
                st.success(f"🟢 LOW RISK OF CHURN — Probability: {prob_pct}")

            st.metric(label="Churn Probability", value=prob_pct)

            # ── Counterfactual interventions (only shown for high-risk) ───────
            # interventions is a list of counterfactual option dicts.
            # Each option contains a list of individual feature changes.
            # We flatten them into a single DataFrame for easy reading.
            interventions = result.get("interventions", [])

            if interventions:
                st.divider()
                st.subheader("🛠️ Recommended Interventions to Prevent Churn")
                st.write(
                    "The model found the following minimum changes that would "
                    "flip this customer's prediction to **low risk**. "
                    "Each option is an independent set of actions."
                )

                # Build a flat table: Option | Feature | Current Value | Suggested Value | Predicted Churn After
                rows = []
                for cf in interventions:
                    cf_id    = cf["counterfactual_id"]
                    cf_proba = cf.get("predicted_churn_proba")
                    cf_prob_str = f"{cf_proba * 100:.1f}%" if cf_proba is not None else "N/A"

                    if not cf.get("interventions"):
                        continue

                    for action in cf["interventions"]:
                        impact_val = action.get("impact", 0)
                        
                        impact_str = f"+{impact_val}%" if impact_val > 0 else f"{impact_val}%"

                        rows.append({
                            "Option":               f"Option {cf_id}",
                            "Feature":              action["feature"].replace("_", " ").title(),
                            "Current Value":        action["original"],
                            "Suggested Value":      action["suggested"],
                            "Impact on Churn":      impact_str,
                            "Churn Prob After":     cf_prob_str,
                        })

                if rows:
                    df_interventions = pd.DataFrame(rows)

                    for option_name, group in df_interventions.groupby("Option"):
                        cf_prob_after = group["Churn Prob After"].iloc[0]
                        with st.expander(
                            f"📋 {option_name}  —  Churn probability drops to {cf_prob_after}",
                            expanded=True
                        ):
                            display_df = group[["Feature", "Current Value", "Suggested Value", "Impact on Churn"]].copy()
                            display_df["Current Value"] = display_df["Current Value"].astype(str)
                            display_df["Suggested Value"] = display_df["Suggested Value"].astype(str)

                            st.dataframe(
                                display_df.reset_index(drop=True),
                                width="stretch",       # Fixed the deprecation warning
                                hide_index=True
                            )
                    st.divider()
                    st.subheader("🤖 AI Agent: Drafting Action Plans...")
                    
                    with st.spinner("Analyzing interventions and drafting communications..."):
                        # We send the original customer data AND the DiCE payload to the agent
                        agent_payload = {
                            "customer_profile": customer_data,
                            "prediction_payload": result 
                        }
                        
                        try:
                            # Call the new LangGraph endpoint
                            AGENT_URL = "http://127.0.0.1:8000/agent/orchestrate"
                            agent_response = requests.post(AGENT_URL, json=agent_payload)
                            
                            if agent_response.status_code == 200:
                                agent_result = agent_response.json()
                                action_plan = agent_result.get("action_plan", {})
                                

                                if "email_draft" in action_plan:
                                    st.info("✉️ **Marketing Route Triggered**")
                                    st.write("Suggested Email Draft to Customer:")
                                    # Create a scrollable box with a set height
                                    with st.container(height=400):
                                        st.markdown(action_plan["email_draft"])
                                
                                # Display the Internal CSM Ticket if it exists
                                if "csm_ticket" in action_plan:
                                    st.warning("🎫 **Support Route Triggered**")
                                    st.write("Internal Brief for Customer Success Manager:")
                                    # Create a scrollable box with a set height
                                    with st.container(height=400):
                                        st.markdown(action_plan["csm_ticket"])
                            else:
                                st.error(f"Agent Error: {agent_response.text}")
                        except requests.exceptions.ConnectionError:
                            st.error("Could not connect to the Agent API.")
                else:
                    st.info("DiCE could not find actionable interventions for this customer profile.")

            elif result["risk_level"] == "High":
                # High risk but no interventions returned (DiCE failed silently)
                st.warning("⚠️ Could not generate intervention recommendations for this customer.")

        else:
            st.warning(f"Error from API: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure your FastAPI server is running!")