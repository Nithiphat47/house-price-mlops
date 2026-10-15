import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/dataset.csv')
df = pd.get_dummies(df, columns=['location'], drop_first=True)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("House_Price_Prediction")

def train_and_log(model, model_name):
    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(
            model, 
            "model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"✅ Model: {model_name: <20} | RMSE: {rmse:,.2f}")
        return mlflow.active_run().info.run_id, rmse, model

print("🚀 เริ่มเทรนโมเดลและบันทึกผลลง MLflow...\n")

run_id_lr, rmse_lr, model_lr = train_and_log(LinearRegression(), "Linear_Regression")
run_id_rf, rmse_rf, model_rf = train_and_log(RandomForestRegressor(n_estimators=100, random_state=42), "Random_Forest")

print("\n🏆 สรุปผลการคัดเลือกโมเดล:")
if rmse_rf < rmse_lr:
    best_run_id = run_id_rf
    best_model = model_rf
    print(f"-> เลือก Random Forest (RMSE ต่ำกว่า)")
else:
    best_run_id = run_id_lr
    best_model = model_lr
    print(f"-> เลือก Linear Regression (RMSE ต่ำกว่า)")

joblib.dump(best_model, "model.pkl")
print("💾 บันทึกไฟล์ model.pkl สำเร็จ!")

model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "house-price-model")
print("✅ ลงทะเบียน house-price-model สำเร็จ!")