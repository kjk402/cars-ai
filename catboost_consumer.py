import threading
import json
import pandas as pd
import numpy as np
import base64
import joblib
from io import BytesIO
from kafka import KafkaConsumer, KafkaProducer
import uuid
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from catboost import CatBoostRegressor

matplotlib.rcParams['font.family'] = 'NanumGothic'

# 경로 설정
model_path = "saved_models/catboost_model.cbm"
scaler_path = "saved_models/scaler_catboost.pkl"
encoders_path = "saved_models/label_encoders_catboost.pkl"
csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"

scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)
model = CatBoostRegressor()
model.load_model(model_path)

def safe_label_transform(le, values):
    classes = set(le.classes_)
    return [le.transform([v])[0] if v in classes else 0 for v in values]

# Kafka 초기화
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def run_prediction_consumer():
    consumer = KafkaConsumer(
        'car-predict-request',
        bootstrap_servers=['kafka:9092'],
        group_id='car-predict-group-catboost',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    print("✅ CatBoost prediction consumer is running...")

    for message in consumer:
        try:
            data = message.value
            request_id = data.pop("request_id", None)
            df_input = pd.DataFrame([data])

            # 파생 변수 생성
            df_input["age"] = 2025 - df_input["year"]
            df_input["mileage_per_year"] = df_input["mileage"] / (df_input["age"] + 1)
            df_input["price_per_cc"] = 0
            for col in ["fuelType", "brand", "model"]:
                df_input[col] = safe_label_transform(label_encoders[col], df_input[col].astype(str))
                df_input[col] = df_input[col].astype(str)

            # 스케일링
            X_num_scaled = scaler.transform(
                df_input[["engineSize", "year", "mileage", "age", "mileage_per_year", "price_per_cc"]])
            X_num_df = pd.DataFrame(X_num_scaled, columns=["engineSize", "year", "mileage", "age", "mileage_per_year",
                                                           "price_per_cc"])
            X_cat_df = df_input[["fuelType", "brand", "model"]].reset_index(drop=True)

            ordered_columns = [
                "engineSize", "year", "mileage",
                "fuelType", "brand", "model",
                "age", "mileage_per_year", "price_per_cc"
            ]
            X_input = pd.concat([X_num_df, X_cat_df], axis=1)[ordered_columns]

            # 예측
            log_pred = model.predict(X_input)[0]
            prediction = np.expm1(log_pred)

            # 그래프 그리기
            df_all = pd.read_csv(csv_path)
            df_all = df_all[df_all["fuelType"].notna() & df_all["price"].notna() & df_all["mileage"].notna()]
            tolerance = 0.1
            filtered = df_all[
                (df_all["year"] == data["year"]) &
                (abs(df_all["engineSize"] - data["engineSize"]) <= tolerance) &
                (df_all["fuelType"] == data["fuelType"])
            ]

            plt.figure(figsize=(8, 6))
            if not filtered.empty:
                sns.regplot(x="mileage", y="price", data=filtered, scatter=False, lowess=True, color="blue", label="가격 추세")
            plt.scatter(filtered["mileage"], filtered["price"], alpha=0.4, label="비슷한 차량", zorder=1)
            plt.scatter(data["mileage"], prediction, color="red", s=100, label="예측 차량", zorder=3)
            plt.plot([data["mileage"], data["mileage"]], [0, prediction], linestyle="--", color="red", zorder=2)

            plt.xlabel("주행거리 (ml)")
            plt.ylabel("price (£)")
            plt.title(f"{data['brand'].upper()} {data['model']} {data['year']}년식 차량 가격 예측")
            plt.legend()
            plt.grid(True)

            buffer = BytesIO()
            plt.savefig(buffer, format="jpeg", dpi=300)
            plt.close()
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            response = {
                "request_id": request_id,
                "predicted_price": int(prediction),
                "image_base64": img_base64,
            }

            producer.send('car-predict-response', value=response)
            print(f"✅ Sent CatBoost prediction for request_id={request_id}")
            print(f"✅ Sent price={int(prediction)}")

        except Exception as e:
            print("❌ Error handling CatBoost prediction message:", e)

# 헬스체크
def run_health_check_consumer():
    consumer = KafkaConsumer(
        'car-health-check-request',
        bootstrap_servers=['kafka:9092'],
        group_id=f'car-health-group-catboost-{uuid.uuid4()}',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    print("✅ Health check consumer (CatBoost) is running...")

    for message in consumer:
        try:
            data = message.value
            request_id = data.get("request_id")

            if data.get("health_check") is True:
                print(f"📡 CatBoost health check ping received (id={request_id})")
                producer.send('car-health-check-response', value={
                    "request_id": request_id,
                    "status": "ok"
                })
        except Exception as e:
            print("❌ Error in CatBoost health check:", e)

if __name__ == "__main__":
    threading.Thread(target=run_prediction_consumer, daemon=True).start()
    threading.Thread(target=run_health_check_consumer, daemon=True).start()

    while True:
        time.sleep(1)
