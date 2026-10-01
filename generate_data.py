import pandas as pd
import numpy as np
import os

# ตรวจสอบและสร้างโฟลเดอร์ data ถ้ายังไม่มี
os.makedirs('data', exist_ok=True)

# ล็อกค่า seed เพื่อให้สุ่มข้อมูลกี่ครั้งก็ได้ผลลัพธ์เดิม
np.random.seed(42)

# จำนวนข้อมูลจำลอง 500 แถว
n_samples = 500

# สร้าง Features
area_sqm = np.random.randint(50, 300, n_samples)       # ขนาดพื้นที่ 50 - 300 ตร.ม.
bedrooms = np.random.randint(1, 6, n_samples)          # จำนวนห้องนอน 1 - 5 ห้อง
age_years = np.random.randint(0, 30, n_samples)        # อายุบ้าน 0 - 30 ปี
location_multiplier = np.random.choice([1.0, 1.2, 1.5], n_samples) # ตัวคูณทำเล

# คำนวณราคา (Price) อิงจากสมการ: พื้นที่ + ห้องนอน - อายุบ้าน + ทำเล + ค่าความคลาดเคลื่อน
base_price = 1000000 + (area_sqm * 50000) + (bedrooms * 200000) - (age_years * 30000)
price = base_price * location_multiplier + np.random.normal(0, 200000, n_samples)

# แปลงตัวคูณทำเลให้เป็นข้อความ (Categorical Data)
location_map = {1.0: 'Rural', 1.2: 'Suburban', 1.5: 'Urban'}
location = [location_map[m] for m in location_multiplier]

# สร้าง DataFrame
df = pd.DataFrame({
    'area_sqm': area_sqm,
    'bedrooms': bedrooms,
    'age_years': age_years,
    'location': location,
    'price': price.astype(int) # ปัดเศษราคาให้เป็นจำนวนเต็ม
})

# บันทึกเป็นไฟล์ CSV
file_path = 'data/dataset.csv'
df.to_csv(file_path, index=False)
print(f"✅ สร้างข้อมูลจำลองจำนวน {n_samples} แถว สำเร็จ! บันทึกไว้ที่ {file_path}")