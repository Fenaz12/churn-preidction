import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

import boto3
from dotenv import load_dotenv

from ml_pipeline.data_processing import ChurnFeatureEngineer

load_dotenv()
bucket_name = os.getenv("S3_BUCKET_NAME")

def train_model():
    print("Loading data...")
    df = pd.read_csv('data/customer_churn_business_dataset.csv')

    df.columns = df.columns.str.replace(' ', '_')
    for col in df.columns:
        if str(df[col].dtype) == 'object':
            df[col] = df[col].str.strip().str.replace(' ', '_', regex=True)


    df["complaint_type"] = df["complaint_type"].fillna("No_Complaint")

    cat_cols = [
        'gender', 'country', 'customer_segment', 'contract_type',
        'payment_method', 'signup_channel', 'discount_applied',
        'price_increase_last_3m', 'survey_response'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )


    dice_cols = [
        'age', 'gender', 'country', 'customer_segment', 'tenure_months',
        'signup_channel', 'contract_type', 'monthly_logins', 'weekly_active_days',
        'avg_session_time', 'features_used', 'usage_growth_rate', 'last_login_days_ago',
        'monthly_fee', 'total_revenue', 'payment_method', 'payment_failures',
        'discount_applied', 'price_increase_last_3m', 'support_tickets',
        'avg_resolution_time', 'complaint_type', 'csat_score', 'escalations',
        'email_open_rate', 'marketing_click_rate', 'nps_score', 'survey_response',
        'referral_count'
    ]
    X_train[[c for c in dice_cols if c in X_train.columns]].to_csv(
        'data/X_train_raw.csv', index=False
    )


    numerical_columns_to_scale = [
        'age', 'tenure_months', 'monthly_fee', 'total_revenue',
        'monthly_logins', 'weekly_active_days', 'avg_session_time',
        'support_tickets', 'escalations', 'avg_resolution_time', 'csat_score',
        'payment_failures', 'last_login_days_ago', 'usage_growth_rate',
        'nps_score', 'email_open_rate', 'marketing_click_rate',
        'features_used', 'referral_count',
        'total_support_wait_time', 'escalation_rate', 'engagement_score'
    ]
    categorical_columns = [
        'gender', 'country', 'customer_segment', 'contract_type',
        'payment_method', 'complaint_type',
        'signup_channel', 'discount_applied', 'price_increase_last_3m', 'survey_response'
    ]
    binary_columns_to_leave_alone = ['unhappy_flag', 'low_engagement_flag']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_columns_to_scale),
            ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))]), categorical_columns),
            ('bin', 'passthrough', binary_columns_to_leave_alone) 
        ])


    pipeline = Pipeline(steps=[
        ('feature_eng', ChurnFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            max_depth=5 ,    #3   
            gamma=0,
            reg_lambda=1, #50
            learning_rate=0.01,   
            n_estimators=300,      
            random_state=42
        ))
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    local_model_path = 'models/churn_model_pipeline.pkl'
    joblib.dump(pipeline, local_model_path)
    print(f"Model saved locally to {local_model_path}")
    
    s3_file_name = "churn_model_pipeline.pkl"
    if bucket_name:
        print(f"Uploading model to S3 bucket: {bucket_name}...")
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_model_path, bucket_name, s3_file_name)
        print("Upload successful!")
    else:
        print("Skipping S3 upload")

if __name__ == "__main__":
    train_model()