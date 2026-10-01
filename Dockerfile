# ใช้ Python เวอร์ชันน้ำหนักเบา
FROM python:3.10-slim

# กำหนดโฟลเดอร์ทำงานใน Container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt และติดตั้งไลบรารี
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโฟลเดอร์โปรเจกต์ทั้งหมด (รวมถึง mlruns ที่เก็บโมเดลไว้) เข้าไปใน Container
COPY . .

# เปิดพอร์ต 8000 สำหรับ FastAPI
EXPOSE 8000

# คำสั่งรัน FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]