import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"
deep_model_path = "saved_models/deep_model.pth"
catboost_model_path = "saved_models/catboost_model.cbm"

# 데이터 로딩 및 전처리
df = pd.read_csv(csv_path)
features = ["engineSize", "year", "mileage", "fuelType", "brand", "model"]
target_col = "price"

df = df[features + [target_col]]
df = df.dropna(subset=[target_col])
df = df.fillna(df.median(numeric_only=True))

label_encoders = joblib.load("saved_models/label_encoders.pkl")
for col in ["fuelType", "brand", "model"]:
    df[col] = label_encoders[col].transform(df[col].astype(str))

# 로그 변환 + 이상치 제거
Q1 = df[target_col].quantile(0.01)
Q3 = df[target_col].quantile(0.99)
df = df[(df[target_col] >= Q1) & (df[target_col] <= Q3)]
df[target_col] = np.log1p(df[target_col])

X = df[features]
y = df[target_col].values
scaler = joblib.load("saved_models/scaler.pkl")
X_num_scaled = scaler.transform(X[["engineSize", "year", "mileage"]])
X_cat = X[["fuelType", "brand", "model"]].values


# PyTorch 모델 정의
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


# 모델 및 예측
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("✅ GPU:", torch.cuda.get_device_name(0))

cat_dims = [len(label_encoders[col].classes_) for col in ["fuelType", "brand", "model"]]
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = DeepCarPriceModel(3, cat_dims, emb_dims).to(device)
model.load_state_dict(torch.load(deep_model_path, map_location=device))
model.eval()

X_num_tensor = torch.tensor(X_num_scaled, dtype=torch.float32).to(device)
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long).to(device)

# PyTorch 예측
start_time = time.time()
with torch.no_grad():
    y_pred_pytorch_log = model(X_num_tensor, X_cat_tensor).cpu().numpy().flatten()
pytorch_time = time.time() - start_time

# CatBoost 예측
catboost_model = CatBoostRegressor()
catboost_model.load_model(catboost_model_path)

X_scaled = pd.DataFrame(X_num_scaled, columns=["engineSize", "year", "mileage"])
X_cat_df = pd.DataFrame(X_cat, columns=["fuelType", "brand", "model"])
X_input_cb = pd.concat([X_scaled, X_cat_df], axis=1)

start_time = time.time()
y_pred_catboost_log = catboost_model.predict(X_input_cb)
catboost_time = time.time() - start_time

# 역변환 및 성능 평가
y_true = np.expm1(y)
y_pred_pytorch = np.expm1(y_pred_pytorch_log)
y_pred_catboost = np.expm1(y_pred_catboost_log)

rmse_pytorch = mean_squared_error(y_true, y_pred_pytorch) ** 0.5
mae_pytorch = mean_absolute_error(y_true, y_pred_pytorch)

rmse_cat = mean_squared_error(y_true, y_pred_catboost) ** 0.5
mae_cat = mean_absolute_error(y_true, y_pred_catboost)

# 결과 출력
print("📊 PyTorch 결과")
print(f"RMSE: {rmse_pytorch:.2f}")
print(f"MAE: {mae_pytorch:.2f}")
print(f"예측 시간: {pytorch_time:.4f}초")

print("\n📊 CatBoost 결과")
print(f"RMSE: {rmse_cat:.2f}")
print(f"MAE: {mae_cat:.2f}")
print(f"예측 시간: {catboost_time:.4f}초")

print("\n📦 모델 크기")
print(f"PyTorch 모델: {os.path.getsize(deep_model_path) / 1024:.2f} KB")
print(f"CatBoost 모델: {os.path.getsize(catboost_model_path) / 1024:.2f} KB")
