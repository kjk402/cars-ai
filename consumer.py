import base64
import json
import threading
import time
import uuid
from io import BytesIO

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer

matplotlib.rcParams['font.family'] = 'NanumGothic'


# 모델
class DeepCarPriceModel(nn.Module):
    def __init__(self, num_numerical, cat_dims, emb_dims):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, emb_dims)
        ])
        emb_total_dim = sum(emb_dims)
        self.model = nn.Sequential(
            nn.Linear(num_numerical + emb_total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_num, x_cat):
        x_cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x_cat_combined = torch.cat(x_cat_embs, dim=1)
        x = torch.cat([x_num, x_cat_combined], dim=1)
        return self.model(x)


# 파일 경로 및 모델 로딩
model_path = "saved_models/deep_model.pth"
scaler_path = "saved_models/scaler.pkl"
encoders_path = "saved_models/label_encoders.pkl"
csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

cat_dims = [len(label_encoders[col].classes_) for col in ["fuelType", "brand", "model"]]
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = DeepCarPriceModel(num_numerical=3, cat_dims=cat_dims, emb_dims=emb_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Kafka  초기화
producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def run_prediction_consumer():
    consumer = KafkaConsumer(
        'car-predict-request',
        bootstrap_servers=['kafka:9092'],
        group_id='car-predict-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    print("✅ Prediction consumer is running...")

    for message in consumer:
        try:
            data = message.value
            request_id = data.pop("request_id", None)
            df_input = pd.DataFrame([data])

            for col in ["fuelType", "brand", "model"]:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

            X_cat = torch.tensor(df_input[["fuelType", "brand", "model"]].values, dtype=torch.long).to(device)
            X_num = torch.tensor(scaler.transform(df_input[["engineSize", "year", "mileage"]]), dtype=torch.float32).to(
                device)

            with torch.no_grad():
                log_pred = model(X_num, X_cat).item()
                prediction = np.expm1(log_pred)

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
                sns.regplot(x="mileage", y="price", data=filtered, scatter=False, lowess=True, color="blue",
                            label="가격 추세")
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
            print(f"✅ Sent prediction for request_id={request_id}")

        except Exception as e:
            print("❌ Error handling prediction message:", e)


# 헬스체크 요청
def run_health_check_consumer():
    consumer = KafkaConsumer(
        'car-health-check-request',
        bootstrap_servers=['kafka:9092'],
        group_id=f'car-health-group-{uuid.uuid4()}',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    print("✅ Health check consumer is running...")

    for message in consumer:
        try:
            data = message.value
            request_id = data.get("request_id")

            if data.get("health_check") is True:
                print(f"📡 Health check ping received (id={request_id})")
                producer.send('car-health-check-response', value={
                    "request_id": request_id,
                    "status": "ok"
                })
        except Exception as e:
            print("❌ Error in health check:", e)


# 스레드로 두 consumer 실행
if __name__ == "__main__":
    threading.Thread(target=run_prediction_consumer, daemon=True).start()
    threading.Thread(target=run_health_check_consumer, daemon=True).start()

    while True:
        time.sleep(1)
