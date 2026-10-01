from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# โหลดโมเดลเวอร์ชันล่าสุดที่เราลงทะเบียนไว้ใน MLflow
MODEL_URI = "models:/house-price-model/1"
model = mlflow.sklearn.load_model(MODEL_URI)

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