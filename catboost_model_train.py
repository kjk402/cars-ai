import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

csv_path = "/mnt/c/Users/joonk/csvs/uk_car.csv"
df = pd.read_csv(csv_path)

features = ["engineSize", "year", "mileage", "fuelType", "brand", "model"]
target_col = "price"

# 전처리
df = df[features + [target_col]]
df = df.dropna(subset=[target_col])
df = df.fillna(df.median(numeric_only=True))

cat_cols = ["fuelType", "brand", "model"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 로그 변환
Q1 = df[target_col].quantile(0.01)
Q3 = df[target_col].quantile(0.99)
df = df[(df[target_col] >= Q1) & (df[target_col] <= Q3)]
df[target_col] = np.log1p(df[target_col])

X = df[features]
y = df[target_col]
scaler = StandardScaler()
X[["engineSize", "year", "mileage"]] = scaler.fit_transform(X[["engineSize", "year", "mileage"]])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost
cat_features_indices = [X.columns.get_loc(col) for col in cat_cols]
train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

# 모델 학습
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=100,
    task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU"
)
model.fit(train_pool, eval_set=val_pool)

# cbm 저장
os.makedirs("saved_models", exist_ok=True)
model.save_model("saved_models/catboost_model.cbm")
joblib.dump(scaler, "saved_models/scaler_catboost.pkl")
joblib.dump(label_encoders, "saved_models/label_encoders_catboost.pkl")

print("✅ CatBoost 모델 저장 완료")
