from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import os

app = FastAPI(title="House Price Prediction API")

# 1. กำหนด URI ของโมเดล (ใช้เวอร์ชัน 1 ตามที่คุณตั้งไว้ หรือเปลี่ยนเป็น "models:/house-price-model/latest" ก็ได้)
MODEL_URI = "models:/house-price-model/1"

# 2. ระบบ Fallback: ลองโหลดจาก MLflow ก่อน หากล้มเหลว (เช่น ตอนอยู่บน Render) ให้โหลดจากไฟล์ model.pkl
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

# กำหนดรูปแบบข้อมูล Input ที่ API จะรับเข้ามา
class HouseFeatures(BaseModel):
    area_sqm: int
    bedrooms: int
    age_years: int
    location: str  # เลือกได้: 'Rural', 'Suburban', 'Urban'

@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API! Go to /docs to test it."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # แปลง location ให้เป็น One-Hot Encoding (1/0) แบบเดียวกับตอน Train
    loc_suburban = 1 if features.location == 'Suburban' else 0
    loc_urban = 1 if features.location == 'Urban' else 0
    
    # สร้าง DataFrame เรียงคอลัมน์ให้ตรงกับตอน Train ข้อมูล
    input_data = pd.DataFrame([{
        'area_sqm': features.area_sqm,
        'bedrooms': features.bedrooms,
        'age_years': features.age_years,
        'location_Suburban': loc_suburban,
        'location_Urban': loc_urban
    }])
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    
    return {
        "predicted_price_thb": round(prediction, 2),
        "input_features": features.dict()
    }