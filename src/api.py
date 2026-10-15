from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import os
from datetime import datetime

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
    location: str

# ----------------------------------------------------
# 📌 ฟีเจอร์ใหม่: Background Task สำหรับเก็บ Log ข้อมูล
# ----------------------------------------------------
def log_prediction_to_csv(input_features: dict, predicted_price: float):
    """
    ฟังก์ชันนี้จะทำงานอยู่เบื้องหลัง เพื่อบันทึกข้อมูลที่ผู้ใช้กรอกเข้ามา
    เอาไว้ใช้เป็น Dataset สำหรับเทรนโมเดลใหม่ในอนาคต (Retraining)
    """
    log_file = "prediction_logs.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode="a", encoding="utf-8") as f:
        # ถ้าไฟล์เพิ่งสร้าง ให้เขียน Header ก่อน
        if not file_exists:
            f.write("timestamp,area_sqm,bedrooms,age_years,location,predicted_price_thb\n")
        
        # บันทึกเวลาและข้อมูลลงไปในไฟล์ CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{input_features['area_sqm']},{input_features['bedrooms']},{input_features['age_years']},{input_features['location']},{predicted_price:.2f}\n")

# ----------------------------------------------------
# 📌 Routes (เส้นทาง API)
# ----------------------------------------------------

@app.get("/", tags=["default"])
def read_root():
    return {"message": "Welcome to House Price Prediction API! Go to /docs to test it."}

@app.get("/health", tags=["default"])
def health_check():
    return {"status": "healthy", "model_loaded": True}

# เพิ่ม background_tasks เข้ามาในพารามิเตอร์
@app.post("/predict", tags=["default"])
def predict_price(features: HouseFeatures, background_tasks: BackgroundTasks):
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
    
    # สั่งให้ฟังก์ชันไปแอบเซฟข้อมูลลงไฟล์ CSV โดยไม่ต้องรอให้เสร็จ (ผู้ใช้จะได้คำตอบทันที)
    background_tasks.add_task(log_prediction_to_csv, features.dict(), prediction)
    
    return {
        "predicted_price_thb": round(prediction, 2),
        "input_features": features.dict()
    }

@app.get("/metrics", tags=["default"])
def get_metrics():
    return {
        "status": "active",
        "description": "Metrics endpoint is ready."
    }