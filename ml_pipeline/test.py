import pandas as pd
import joblib
import numpy as np
from ml_pipeline.counterfactuals import (
    load_explainer, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
    ACTIONABLE_FEATURES, FEATURE_RANGES
)

# ─────────────────────────────────────────────────────────────
# LAYER 1: Does the model actually think this customer is high risk?
# If churn probability is low, DiCE has nothing to flip.
# ─────────────────────────────────────────────────────────────
model = joblib.load("models/churn_model_pipeline.pkl")

customer = pd.DataFrame([{
    "age": 30, "gender": "Male", "country": "Bangladesh",
    "customer_segment": "Regular", "tenure_months": 12,
    "contract_type": "Monthly", "monthly_fee": 50, 
    "total_revenue": 600, "monthly_logins": 15,
    "weekly_active_days": 3, "avg_session_time": 25.0,
    "support_tickets": 1, "escalations": 0,
    "avg_resolution_time": 12.0, "csat_score": 1.0, 
    "payment_method": "Card", "complaint_type": "No_Complaint",
    "payment_failures": 0, "last_login_days_ago": 5,
    "usage_growth_rate": 0, "nps_score": 20,
    "email_open_rate": 0.5, "marketing_click_rate": 0.25,
    "features_used": 3, "referral_count": 0,
    "signup_channel": "Web", "discount_applied": "No",
    "price_increase_last_3m": "No", "survey_response": "Satisfied"
}])

proba = model.predict_proba(customer)[0][1]
print(f"[LAYER 1] Churn probability: {proba:.4f}")

# Update threshold to match our new business logic
threshold = 0.15  
if proba < threshold:
    print(f"  ❌ PROBLEM HERE: Model thinks this is LOW RISK (< {threshold}).")
    import sys
    sys.exit() # <-- Forces the script to stop so we don't feed DiCE garbage!
else:
    print("  ✅ Model predicts HIGH RISK — DiCE has something to flip.")

print()

# ─────────────────────────────────────────────────────────────
# LAYER 2: Does DiCE actually return a dataframe with rows?
# If final_cfs_df is None or empty, DiCE failed internally.
# ─────────────────────────────────────────────────────────────
explainer = load_explainer(
    model_path="models/churn_model_pipeline.pkl",
    train_data_path="data/X_train_raw.csv"
)

raw_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
query = customer[raw_cols].copy()

try:
    cf_result = explainer.dice_exp.generate_counterfactuals(
            query_instances=query,
            total_CFs=3,
            desired_class=0,
            permitted_range=FEATURE_RANGES,
            features_to_vary=ACTIONABLE_FEATURES,
            proximity_weight=1.5,  # <--- Allows larger magnitude changes
            diversity_weight=1,  
            verbose=True,   
        )
    cf_examples = cf_result.cf_examples_list[0]
    cf_df = cf_examples.final_cfs_df

    print(f"[LAYER 2] DiCE returned: {type(cf_df)}")
    if cf_df is None:
        print("  ❌ PROBLEM HERE: cf_df is None.")
        print("     Fix: DiCE couldn't find any valid counterfactuals in the search space.")
        print("     Try: increase total_CFs to 10, or lower proximity_weight to 0.1")
    elif cf_df.empty:
        print("  ❌ PROBLEM HERE: cf_df is an empty DataFrame.")
    else:
        print(f"  ✅ DiCE returned {len(cf_df)} rows")
        print(f"     Columns: {cf_df.columns.tolist()}")

except Exception as e:
    print(f"  ❌ PROBLEM HERE: DiCE raised an exception: {e}")
    cf_df = None

print()

# ─────────────────────────────────────────────────────────────
# LAYER 3: Do the column names in cf_df match ACTIONABLE_FEATURES?
# If DiCE renamed/dropped columns, our diff loop finds nothing.
# ─────────────────────────────────────────────────────────────
if cf_df is not None and not cf_df.empty:
    missing_in_cf   = [f for f in ACTIONABLE_FEATURES if f not in cf_df.columns]
    missing_in_orig = [f for f in ACTIONABLE_FEATURES if f not in query.columns]

    print(f"[LAYER 3] ACTIONABLE_FEATURES missing from DiCE output: {missing_in_cf}")
    print(f"          ACTIONABLE_FEATURES missing from original query: {missing_in_orig}")

    if missing_in_cf:
        print("  ❌ PROBLEM HERE: DiCE dropped or renamed some columns.")
        print(f"     Actual cf_df columns: {cf_df.columns.tolist()}")
        print("     Fix: filter cf_df to only use columns that exist in both.")
    else:
        print("  ✅ All ACTIONABLE_FEATURES present in DiCE output")

    print()

    # ─────────────────────────────────────────────────────────
    # LAYER 4: Are the values actually different from the original?
    # If every diff is < 0.01, nothing gets added to interventions.
    # ─────────────────────────────────────────────────────────
    print("[LAYER 4] Feature-by-feature diff (original vs first counterfactual):")
    original_row = query.iloc[0]
    cf_row = cf_df.iloc[0]

    changed = []
    unchanged = []

    for feature in ACTIONABLE_FEATURES:
        if feature not in cf_row.index or feature not in original_row.index:
            continue
        orig_val = original_row[feature]
        new_val  = cf_row[feature]
        try:
            diff = abs(float(orig_val) - float(new_val))
            if diff >= 0.01:
                changed.append(f"  CHANGED  {feature}: {orig_val} → {new_val}  (diff={diff:.4f})")
            else:
                unchanged.append(f"  same     {feature}: {orig_val}")
        except (TypeError, ValueError):
            if str(orig_val).strip() != str(new_val).strip():
                changed.append(f"  CHANGED  {feature}: '{orig_val}' → '{new_val}'")
            else:
                unchanged.append(f"  same     {feature}: '{orig_val}'")

    if changed:
        print(f"  ✅ {len(changed)} features changed:")
        for c in changed:
            print(c)
    else:
        print("  ❌ PROBLEM HERE: Zero features changed between original and counterfactual.")
        print("     This means DiCE returned the same values as the input — its search failed.")
        print("     Fix: set verbose=True in generate_counterfactuals and read DiCE's output.")

    print(f"\n  (Unchanged: {len(unchanged)} features — these are fine)")


