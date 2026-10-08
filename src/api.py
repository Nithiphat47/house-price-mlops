from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import os

app = FastAPI(title="House Price Prediction API", version="1.0.0")

MODEL_URI = "models:/house-price-model/1"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Loaded model from MLflow Registry")
except Exception as e:
    print(f"⚠️ MLflow artifact not found. Falling back to local file: {e}")
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        print("✅ Loaded model directly from model.pkl")
    else:
        raise RuntimeError("❌ Model file not found! Please run train.py first.")

class HouseFeatures(BaseModel):
    area_sqm: int
    bedrooms: int
    age_years: int
    location: str  # เลือกได้: 'Rural', 'Suburban', 'Urban'

# ----------------------------------------------------
# 📌 Routes (เส้นทาง API)
# ----------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API! Go to /docs to test it."}

# [เพิ่มใหม่] 1. Health Check
@app.get("/health", tags=["default"])
def health_check():
    return {"status": "healthy", "model_loaded": True}

# 2. Predict Price (ของเดิม)
@app.post("/predict", tags=["default"])
def predict_price(features: HouseFeatures):
    loc_suburban = 1 if features.location == 'Suburban' else 0
    loc_urban = 1 if features.location == 'Urban' else 0
    
    input_data = pd.DataFrame([{
        'area_sqm': features.area_sqm,
        'bedrooms': features.bedrooms,
        'age_years': features.age_years,
        'location_Suburban': loc_suburban,
        'location_Urban': loc_urban
    }])
    
    prediction = model.predict(input_data)[0]
    
    return {
        "predicted_price_thb": round(prediction, 2),
        "input_features": features.dict()
    }

# [เพิ่มใหม่] 3. Metrics
@app.get("/metrics", tags=["default"])
def get_metrics():
    # ส่งค่าสถิติจำลองกลับไป (พร้อมต่อยอดกับ Prometheus ในอนาคต)
    return {
        "status": "active",
        "description": "Metrics endpoint is ready."
    }