from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import os

# ----------------------------------------------------
# 📌 ตกแต่งหน้า Swagger UI ด้วย Markdown & Emojis
# ----------------------------------------------------
description_text = """
## 🌟 ยินดีต้อนรับสู่โปรเจกต์ทำนายราคาบ้าน
โปรเจกต์นี้เป็นส่วนหนึ่งของการนำโมเดล Machine Learning ขึ้นระบบคลาวด์ (MLOps)

### 🚀 ฟีเจอร์การใช้งาน:
* 🏠 **ทำนายราคา (Predict):** ป้อนขนาดพื้นที่, จำนวนห้องนอน, อายุบ้าน และทำเลที่ตั้ง
* 🩺 **สถานะระบบ (Health):** ตรวจสอบการทำงานของเซิร์ฟเวอร์
* 📊 **สถิติ (Metrics):** ดูข้อมูลการใช้งาน API เบื้องต้น

👨‍💻 **พัฒนาโดย:** Nithiphat (Bomb) - Computer Engineering
"""

tags_meta = [
    {"name": "Predict", "description": "ส่วนสำหรับกรอกข้อมูลเพื่อประเมินราคาบ้าน 💸"},
    {"name": "System", "description": "ส่วนสำหรับมอนิเตอร์และตรวจสอบสถานะเซิร์ฟเวอร์ ⚙️"}
]

app = FastAPI(
    title="🏡 House Price Prediction API",
    description=description_text,
    version="1.0.0",
    openapi_tags=tags_meta,
    contact={
        "name": "Nithiphat Sinthong",
        "url": "https://github.com/Nithiphat47/house-price-mlops",
    }
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

class HouseFeatures(BaseModel):
    area_sqm: int
    bedrooms: int
    age_years: int
    location: str  # เลือกได้: 'Rural', 'Suburban', 'Urban'

# ----------------------------------------------------
# 📌 Routes (เส้นทาง API)
# ----------------------------------------------------

@app.get("/", tags=["System"])
def read_root():
    return {"message": "Welcome to House Price Prediction API! Go to /docs to test it."}

# [เพิ่มใหม่] 1. Health Check
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model_loaded": True}

# 2. Predict Price (ของเดิม)
@app.post("/predict", tags=["Predict"])
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
@app.get("/metrics", tags=["System"])
def get_metrics():
    # ส่งค่าสถิติจำลองกลับไป (พร้อมต่อยอดกับ Prometheus ในอนาคต)
    return {
        "status": "active",
        "description": "Metrics endpoint is ready."
    }