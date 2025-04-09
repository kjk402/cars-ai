import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import seaborn as sns
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'NanumGothic'

class CarFeatures(BaseModel):
    engineSize: float
    year: int
    mileage: int
    fuelType: str
    brand: str
    model: str

class PricePrediction(BaseModel):
    predicted_price: int
    image_base64: str

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if device.type == "cuda":
    print("✅ GPU:", torch.cuda.get_device_name(0))

model_path = "saved_models/deep_model.pth"
scaler_path = "saved_models/scaler.pkl"
encoders_path = "saved_models/label_encoders.pkl"
csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"

scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

cat_dims = [len(label_encoders[col].classes_) for col in ["fuelType", "brand", "model"]]
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = DeepCarPriceModel(num_numerical=3, cat_dims=cat_dims, emb_dims=emb_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

app = FastAPI()

@app.post("/predict", response_model=PricePrediction)
def predict_price(data: CarFeatures):
    df_input = pd.DataFrame([data.dict()])
    try:
        cat_tensor = []
        for col in ["fuelType", "brand", "model"]:
            df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            cat_tensor.append(torch.tensor(df_input[col].values, dtype=torch.long))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Label encoding 실패: {e}")

    X_cat = torch.stack(cat_tensor, dim=1).to(device)
    X_num = df_input[["engineSize", "year", "mileage"]]
    X_num_scaled = scaler.transform(X_num)
    X_num_tensor = torch.tensor(X_num_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_pred = model(X_num_tensor, X_cat).item()
        prediction = np.expm1(log_pred)

    df_all = pd.read_csv(csv_path)
    df_all = df_all[df_all["fuelType"].notna() & df_all["price"].notna() & df_all["mileage"].notna()]
    tolerance = 0.1
    filtered = df_all[
        (df_all["year"] == data.year) &
        (abs(df_all["engineSize"] - data.engineSize) <= tolerance) &
        (df_all["fuelType"] == data.fuelType)
    ]

    plt.figure(figsize=(8, 6))
    if not filtered.empty:
        sns.regplot(x="mileage", y="price", data=filtered, scatter=False, lowess=True, color="blue", label="가격 추세")

    plt.scatter(data.mileage, prediction, color="red", s=100, label="예측 차량", zorder=3)
    plt.scatter(filtered["mileage"], filtered["price"], alpha=0.4, label="비슷한 차량", zorder=1)
    plt.plot([data.mileage, data.mileage], [0, prediction], linestyle="--", color="red", zorder=2)
    plt.xlabel("주행거리 (ml)")
    plt.ylabel("price (£)")
    plt.title(f"{data.brand.upper()} {data.model} {data.year}년식 차량 가격 예측")
    plt.legend()
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format="jpeg", dpi=300)
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return PricePrediction(predicted_price=int(prediction), image_base64=img_base64)
