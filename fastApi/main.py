import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# 2. Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development; update with Streamlit URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Dynamic Path Loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_model.joblib")
TREE_MODEL_PATH = os.path.join(BASE_DIR, "models", "decision_tree_model.joblib")

# Load the models
try:
    log_reg = joblib.load(LOG_MODEL_PATH)
    dec_tree = joblib.load(TREE_MODEL_PATH)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# 4. Define the Data Schema
class CustomerData(BaseModel):
    Usage_Hours: int
    Subscription_Type: str
    Age: int
    Support_Calls: int
    Gender: str

# 5. API Endpoints
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running. Go to /docs for Swagger UI."}

@app.post("/predict/{model_type}")
def predict(model_type: str, data: CustomerData):
    # Convert incoming JSON data to a Pandas DataFrame for the model pipeline
    input_df = pd.DataFrame([data.dict()])
    
    if model_type == "logistic":
        prediction = log_reg.predict(input_df)
    elif model_type == "decision_tree":
        prediction = dec_tree.predict(input_df)
    else:
        return {"error": "Invalid model type. Use 'logistic' or 'decision_tree'."}
        
    return {
        "prediction": int(prediction[0]), 
        "model_used": model_type,
        "status": "Success"
    }