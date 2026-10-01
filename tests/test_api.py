from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_predict_price():
    # จำลองข้อมูลที่ส่งเข้ามาทำนาย
    payload = {
        "area_sqm": 150,
        "bedrooms": 3,
        "age_years": 5,
        "location": "Urban"
    }
    response = client.post("/predict", json=payload)
    
    # ตรวจสอบว่า API ตอบกลับสำเร็จ (200) และมีคีย์ผลลัพธ์ที่ต้องการ
    assert response.status_code == 200
    assert "predicted_price_thb" in response.json()
    assert response.json()["predicted_price_thb"] > 0