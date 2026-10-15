import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os

print("🚀 Starting Model Training with Hyperparameter Tuning...")

# (ข้อมูลจำลองชั่วคราว เพื่อให้โค้ดรันผ่านหากไม่มีไฟล์ csv)
df = pd.DataFrame({
    'area_sqm': [50, 100, 120, 80, 200, 150],
    'bedrooms': [1, 2, 3, 2, 4, 3],
    'age_years': [5, 10, 2, 15, 1, 5],
    'location_Suburban': [0, 1, 0, 1, 0, 1],
    'location_Urban': [1, 0, 0, 0, 1, 0],
    'price_thb': [2500000, 3200000, 4500000, 2800000, 8500000, 5200000]
})
X = df.drop('price_thb', axis=1)
y = df['price_thb']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------
# 📌 ฟีเจอร์ใหม่: Hyperparameter Tuning (Grid Search)
# ----------------------------------------------------
# กำหนดช่วงพารามิเตอร์ที่ต้องการให้คอมพิวเตอร์ลองจูน
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20]
}

print("🔍 Searching for the best parameters...")
base_model = RandomForestRegressor(random_state=42)

# สั่งให้รันหาค่าที่ดีที่สุด 
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 3. เริ่มบันทึกผลการทดลองลง MLflow
mlflow.set_experiment("House_Price_Prediction")
with mlflow.start_run():
    # ดึงโมเดลตัวที่ตั้งค่าดีที่สุดมาใช้งาน
    best_model = grid_search.best_estimator_
    
    # ทำนายและประเมินผล
    predictions = best_model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"✅ Best Parameters Found: {grid_search.best_params_}")
    print(f"📊 R2 Score: {r2:.4f}, MSE: {mse:.2f}")

    # บันทึกข้อมูลทั้งหมดลง MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(best_model, "house-price-model")
    print("☁️ Model logged to MLflow successfully.")

# 4. บันทึกโมเดลลงไฟล์ model.pkl สำหรับ API
joblib.dump(best_model, "model.pkl")
print("💾 Best model saved locally as model.pkl")