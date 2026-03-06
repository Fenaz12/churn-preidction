import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.login_median_ = None
        self.weekly_active_median_ = None
        self.session_time_median_ = None
        
    def fit(self, X, y=None):
        self.login_median_ = X["monthly_logins"].median()
        self.weekly_active_median_ = X["weekly_active_days"].median()
        self.session_time_median_ = X["avg_session_time"].median()
        return self
        
    def transform(self, X):
        X_new = X.copy()
        
        # 1. Total Support Wait Time
        X_new['total_support_wait_time'] = X_new['support_tickets'] * X_new['avg_resolution_time']
        
        # 2. Support Rate & Unhappy Flag
        X_new["unhappy_flag"] = (X_new["csat_score"] <= 3).astype(int)
        X_new["escalation_rate"] = (X_new["support_tickets"] + X_new["escalations"]) / (X_new["tenure_months"] + 1)
        
        # 3. Engagement Score 
        score = (
            (X_new["monthly_logins"] >= self.login_median_).astype(int) +
            (X_new["weekly_active_days"] >= self.weekly_active_median_).astype(int) +
            (X_new["avg_session_time"] >= self.session_time_median_).astype(int)
        )
        X_new["engagement_score"] = score
        
        # 4. Low Engagement Flag 
        X_new["low_engagement_flag"] = (X_new["engagement_score"] < 2).astype(int)
        
        # 5. Fill Missing Complaints 
        X_new["complaint_type"] = X_new["complaint_type"].fillna("No_Complaint")
        
        # 6. Drop Customer ID if it exists
        if 'customer_id' in X_new.columns:
            X_new = X_new.drop(columns=['customer_id'])
            
        return X_new