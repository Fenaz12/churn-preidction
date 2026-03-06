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
git clone https://github.com/Fenaz12/churn-preidction.git
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

While this project runs locally for demonstration, a production-grade deployment would utilize standard cloud architecture principles to scale securely and efficiently:

### 1. Scaling to 100k+ Records (Serving Layer)
* **Containerization:** The FastAPI application would be packaged into a Docker container. This ensures the app runs exactly the same way in the cloud as it does on a local laptop.
* **Horizontal Scaling:** The Docker container would be deployed to a managed cloud environment. We would place a **Load Balancer** in front of the application. As API traffic increases (e.g., processing 100k+ requests), the load balancer automatically spins up duplicate copies of the container to share the workload.

### 2. Model Retraining Strategy
* **Storage:** The `train.py` script currently saves the serialized `.pkl` pipeline directly to cloud object storage (Amazon S3).
* **Automation:** A scheduled automated job (e.g., a Cron job running monthly) would pull fresh customer data from the data warehouse, execute the training pipeline, and overwrite the model in S3. 
* **Seamless Updates:** The FastAPI application would be configured to periodically check the S3 bucket. If it detects a newer version of the `.pkl` file, it will gracefully reload the model into memory without dropping active API requests.

### 3. Monitoring & Logging
* **Infrastructure Metrics:** Standard cloud monitoring tools would track API latency (speed), error rates (HTTP 500s), and server memory usage.
* **Data Drift Detection:** Incoming API payloads from the real world would be logged to a database. A scheduled script would compare the statistical distribution of these incoming requests against our original training dataset. If real-world customer behavior starts shifting (e.g., if the average `monthly_fee` suddenly drops), the system alerts the data science team that it is time to retrain the model.

### 4. Cloud Cost Considerations
* **Storage Costs:** Storing serialized `.pkl` files in object storage like S3 is highly cost-effective (fractions of a cent per month).
* **Compute Costs:** By utilizing an "auto-scaling" or "serverless" compute architecture, the business only pays for server time when API traffic is high. During the night or low-traffic periods, the servers scale down, avoiding the wasted sunk costs of running servers 24/7.

---

## 🎥 Video Demonstration
https://drive.google.com/file/d/1U2YAHYbBpuiFXLH0oGqla4s7K8xBGaPy/view?usp=sharing
