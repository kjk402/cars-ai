import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split

csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"

df = pd.read_csv(csv_path)

features = ["engineSize", "year", "mileage", "fuelType", "brand", "model"]
target_col = "price"

# 전처리
df = df[features + [target_col]]
df = df.dropna(subset=[target_col])
df = df.fillna(df.median(numeric_only=True))

label_encoders = {}
for col in ["fuelType", "brand", "model"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 로그 변환 + 이상치 제거
Q1 = df[target_col].quantile(0.01)
Q3 = df[target_col].quantile(0.99)
df = df[(df[target_col] >= Q1) & (df[target_col] <= Q3)]
y = np.log1p(df[target_col].values.reshape(-1, 1))

X_cat = df[["fuelType", "brand", "model"]].values
X_num = df[["engineSize", "year", "mileage"]]

# 스케일링
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 텐서 변환
X_num_tensor = torch.tensor(X_num_scaled, dtype=torch.float32)
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 데이터셋 생성
dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)


# 모델 정의
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
            nn.Dropout(0.5),  # ✅ Dropout 과적합 방지
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # ✅ Dropout 과적합 방지
            nn.Linear(64, 1)
        )

    def forward(self, x_num, x_cat):
        x_cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x_cat_combined = torch.cat(x_cat_embs, dim=1)
        x = torch.cat([x_num, x_cat_combined], dim=1)
        return self.model(x)


cat_dims = [len(label_encoders[col].classes_) for col in ["fuelType", "brand", "model"]]
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if device.type == "cuda":
    print("✅ GPU:", torch.cuda.get_device_name(0))

model = DeepCarPriceModel(num_numerical=3, cat_dims=cat_dims, emb_dims=emb_dims).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프 - epochs는 로스에 따라 조절
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_num_batch, x_cat_batch, y_batch in train_loader:
        x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_num_batch, x_cat_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"[{epoch + 1}/{epochs}] Train Loss: {train_loss / len(train_loader):.4f}")

# 저장
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/deep_model.pth")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(label_encoders, "saved_models/label_encoders.pkl")

print("✅ 모델과 전처리기 저장 완료.")
