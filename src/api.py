from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal  # นำเข้าตัวช่วยสำหรับบังคับคำ
import mlflow.sklearn
import pandas as pd
import joblib
import os

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# 1. เพิ่ม CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้ทุกโดเมนเชื่อมต่อได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 2. เพิ่ม Data Validation แบบสมบูรณ์
class HouseFeatures(BaseModel):
    area_sqm: int = Field(..., gt=0, description="ขนาดพื้นที่ต้องมากกว่า 0")
    bedrooms: int = Field(..., gt=0, description="จำนวนห้องนอนต้องมากกว่า 0")
    age_years: int = Field(..., ge=0, description="อายุบ้านห้ามติดลบ")
    # ใช้ Literal บังคับให้กรอกได้แค่ 3 ค่านี้เท่านั้น ถ้าพิมพ์อย่างอื่นระบบจะดีดกลับทันที
    location: Literal['Rural', 'Suburban', 'Urban'] = Field(..., description="พิมพ์ 'Rural', 'Suburban' หรือ 'Urban'")

@app.get("/", tags=["default"])
def read_root():
    return {"message": "Welcome to House Price Prediction API! Go to /docs to test it."}

@app.get("/health", tags=["default"])
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", tags=["default"])
def predict_price(features: HouseFeatures):
    # 3. เพิ่ม Error Handling
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

@app.get("/metrics", tags=["default"])
def get_metrics():
    return {
        "status": "active",
        "description": "Metrics endpoint is ready."
    }