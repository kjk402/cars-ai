import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

input_data = {
    "engineSize": 3.0,
    "year": 2017,
    "mileage": 21000,
    "fuelType": "Diesel",
    "brand": "audi",
    "model": " A7"
}

model_path = "saved_models/deep_model.pth"
scaler_path = "saved_models/scaler.pkl"
encoders_path = "saved_models/label_encoders.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

df_input = pd.DataFrame([input_data])
for col in ["fuelType", "brand", "model"]:
    df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

X_cat = df_input[["fuelType", "brand", "model"]].values
X_num = df_input[["engineSize", "year", "mileage"]].values
X_num_scaled = scaler.transform(pd.DataFrame(X_num, columns=["engineSize", "year", "mileage"]))

X_cat_tensor = torch.tensor(X_cat, dtype=torch.long).to(device)
X_num_tensor = torch.tensor(X_num_scaled, dtype=torch.float32).to(device)


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


cat_dims = [len(label_encoders[col].classes_) for col in ["fuelType", "brand", "model"]]
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = DeepCarPriceModel(num_numerical=3, cat_dims=cat_dims, emb_dims=emb_dims).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    log_price = model(X_num_tensor, X_cat_tensor).item()
    predicted_price = np.expm1(log_price)
    print(f"예상 차량 가격: {int(predicted_price):,} 파운드")
