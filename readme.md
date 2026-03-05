# 📉 Customer Churn Prediction System

An end-to-end Machine Learning solution designed to predict customer churn for a SaaS business. This project demonstrates a production-ready AI lifecycle, spanning from Exploratory Data Analysis (EDA) and custom Scikit-Learn pipeline engineering to modular FastAPI deployment, interactive Streamlit dashboards, and AWS S3 cloud integration.

---

## 🚀 Key Features

- **Modular ML Pipeline:** Utilizes custom Scikit-Learn Transformers (`ChurnFeatureEngineer`) embedded directly into the training pipeline to eliminate training-serving skew and maintain state (e.g., historical medians) automatically.
- **Optimized XGBoost Model:** Hyperparameter-tuned via GridSearchCV to balance precision (reducing false alarms/wasted retention budget) and recall (catching at-risk customers).
- **Decoupled API Architecture:** FastAPI routing is heavily modularized (`api/endpoints/predict.py`) and completely decoupled from model training, mirroring enterprise-grade microservice structures.
- **Cloud Integration (AWS S3):** Automatically pushes serialized `.pkl` models to AWS S3 post-training and retrieves them dynamically upon API startup.
- **Interactive UI:** Built-in Streamlit frontend for rapid testing, stakeholder demonstration, and payload generation.

---

## 📂 Project Structure

```text
churn-prediction-project/
│
├── api/
│   ├── __init__.py
│   └── endpoints/
│       ├── __init__.py
│       └── predict.py            # FastAPI router for inference logic
│
├── data/
│   └── customer_churn_business_dataset.csv  # Raw training dataset
│
├── models/
│   └── churn_model_pipeline.pkl  # Serialized Pipeline (Downloaded from S3 automatically)
│
├── ChurnPrediction.ipynb         # EDA, Model Selection, and Business Impact Analysis
├── data_processing.py            # Custom Scikit-Learn Transformer class
├── train.py                      # Model training & AWS S3 upload script
├── main.py                       # FastAPI application runner
├── streamlit_app.py              # Interactive UI dashboard
├── requirements.txt              # Project dependencies
└── .env                          # Environment variables (Ignored in Git)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository and navigate to the directory

```bash
git clone https://github.com/YourUsername/churn-prediction-project.git
cd churn-prediction-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**On Windows**

```bash
.\.venv\Scripts\activate
```

**On Mac/Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory. Ensure this file is added to your `.gitignore` to prevent leaking AWS credentials.

```env
MODEL_PATH=models/churn_model_pipeline.pkl
S3_BUCKET_NAME=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

---

## 💻 Usage Instructions

### 1. Train the Model & Push to Cloud (Optional)

To retrain the model on the dataset, optimize hyperparameters, and upload the new pipeline version to AWS S3:

```bash
python train.py
```

### 2. Start the FastAPI Server

The server will automatically connect to AWS, download the latest model from S3 upon startup, and expose the inference endpoints.

```bash
uvicorn main:app --reload
```

**Interactive API Docs (Swagger UI):**  
Navigate to `http://127.0.0.1:8000/docs` to test the API directly from your browser.

---

### 3. Launch the Streamlit Dashboard

Open a separate terminal window (ensure your virtual environment is still active) and run:

```bash
streamlit run streamlit_app.py
```

This launches a user-friendly web interface that communicates seamlessly with your local FastAPI backend.

---

## 📊 Model Selection & Business Impact

During the EDA phase (detailed in `ChurnPrediction.ipynb`), a baseline Logistic Regression model was evaluated against an XGBoost classifier.

### Why XGBoost?

**Multicollinearity Handling**  
Feature engineering revealed highly correlated features (e.g., raw CSAT scores and the engineered `unhappy_flag`). XGBoost handles this non-linear, correlated data natively, whereas Logistic Regression suffered from wide nets and low precision.

**Business Cost Optimization**  
The optimized XGBoost model prioritized a high F1-score and Precision. In a business context, this minimizes the cost of **False Positives** (wasting $50 retention campaigns on customers who were never going to leave) while maintaining a high ROC-AUC to accurately identify true churners.

This translated to a significantly higher **net savings for the business** compared to the baseline model.

---

## ☁️ Optimization, Scaling & Cloud Deployment Plan

While this project runs locally for demonstration, a production-grade deployment would utilize the following AWS architecture to scale securely and efficiently.

### 1. Scaling to 100k+ Records (Serving Layer)

- **Containerization:** The modular FastAPI application (`main.py` + `api/`) would be containerized using Docker.
- **Hosting:** The Docker image would be deployed to **AWS ECS (Elastic Container Service)** using **AWS Fargate** for serverless compute.
- **Load Balancing:** An **Application Load Balancer (ALB)** would distribute incoming JSON payloads across multiple API instances. As API traffic spikes, **ECS Auto-Scaling** would spin up additional containers dynamically.

### 2. Model Retraining Strategy

- **Storage:** The `train.py` script currently saves the serialized `.pkl` pipeline directly to **Amazon S3**.
- **Automation:** An **AWS EventBridge** cron schedule would trigger an **AWS Step Function** on a weekly/monthly cadence. This function would pull fresh data from the warehouse, execute the training pipeline, and overwrite the model in S3.
- **Zero-Downtime Updates:** The FastAPI application would be configured to poll S3 for object hash changes or receive an **SNS notification** to gracefully reload the `.pkl` file into memory without dropping active HTTP requests.

### 3. Monitoring & Logging

- **Infrastructure Metrics:** **AWS CloudWatch** will track API latency, HTTP 5xx errors, and container CPU/Memory utilization.
- **Data Drift Detection:** Incoming API payloads would be asynchronously logged to a separate S3 bucket via **Amazon Kinesis Firehose**. A daily batch script would compare the statistical distribution of these incoming live requests against our training dataset to detect **Data Drift** (e.g., alerting if the average `monthly_fee` suddenly deviates in the real world).

### 4. Cloud Cost Considerations

- **S3 Storage:** Storing serialized `.pkl` files costs fractions of a cent (~$0.023 per GB/month).
- **Compute (ECS Fargate):** By utilizing serverless containers that scale down during low traffic periods, compute costs remain directly proportional to API usage, avoiding the sunk costs of idle EC2 instances.

---

## 🎥 Video Demonstration

[Link to your 10–15 minute screen recording demonstrating dataset overview, training, API execution, Streamlit UI, and design decisions]