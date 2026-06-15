import boto3
import time

REGION = "ap-southeast-1"
ACCOUNT_ID = "664926621258"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/churn-ml-endpoint:latest"


ROLE_ARN = "arn:aws:iam::664926621258:role/SageMaker-ExecutionRole-ChurnApp"

MODEL_NAME = "churn-prediction-model"
ENDPOINT_CONFIG_NAME = "churn-serverless-config"
ENDPOINT_NAME = "churn-serverless-endpoint"

sm_client = boto3.client("sagemaker", region_name=REGION)

def deploy_serverless_endpoint():
    # 1. Create the Model Entity in SageMaker
    print("1. Creating SageMaker Model...")
    try:
        sm_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={"Image": IMAGE_URI},
            ExecutionRoleArn=ROLE_ARN
        )
        print("   Model created successfully.")
    except Exception as e:
        print(f"   Note: {e}")

    # 2. Create the Serverless Endpoint Configuration (CORRECTED)
    print("\n2. Creating Serverless Endpoint Configuration...")
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": MODEL_NAME,
                    "ServerlessConfig": {
                        "MemorySizeInMB": 2048,  # 2GB RAM
                        "MaxConcurrency": 5      # 5 concurrent requests
                    }
                }
            ]
        )
        print("   Config created successfully.")
    except Exception as e:
        print(f"   Note: {e}")

    # 3. Launch the Endpoint
    print("\n3. Launching Serverless Endpoint...")
    try:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print("\n🚀 Deployment initiated! This usually takes 3 to 5 minutes.")
        print("You can check the exact status in the AWS Console under Amazon SageMaker -> Endpoints.")
    except Exception as e:
        print(f"\n❌ Error deploying endpoint: {e}")

if __name__ == "__main__":
    deploy_serverless_endpoint()