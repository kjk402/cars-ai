import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

model_path = "saved_models/catboost_model.cbm"
scaler_path = "saved_models/scaler_catboost.pkl"
encoders_path = "saved_models/label_encoders_catboost.pkl"

input_data = {
    "engineSize": 3.0,
    "year": 2017,
    "mileage": 21000,
    "fuelType": "Diesel",
    "brand": "audi",
    "model": " A7"
}

scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)
model = CatBoostRegressor()
model.load_model(model_path)

df_input = pd.DataFrame([input_data])

df_input["age"] = 2025 - df_input["year"]
df_input["mileage_per_year"] = df_input["mileage"] / (df_input["age"] + 1)
df_input["price_per_cc"] = 0

for col in ["fuelType", "brand", "model"]:
    df_input[col] = df_input[col].astype(str).str.strip()
    if df_input[col].iloc[0] in label_encoders[col].classes_:
        df_input[col] = label_encoders[col].transform(df_input[col])
    else:
        df_input[col] = 0
    df_input[col] = df_input[col].astype(str)

X_num_scaled = scaler.transform(df_input[["engineSize", "year", "mileage"]])
X_num_df = pd.DataFrame(X_num_scaled, columns=["engineSize", "year", "mileage"])

X_derived_df = df_input[["age", "mileage_per_year", "price_per_cc"]].reset_index(drop=True)
X_cat_df = df_input[["fuelType", "brand", "model"]].reset_index(drop=True)

X_input = pd.concat([X_num_df, X_cat_df, X_derived_df], axis=1)[[
    "engineSize", "year", "mileage",
    "fuelType", "brand", "model",
    "age", "mileage_per_year", "price_per_cc"
]]

log_price = model.predict(X_input)[0]
price = np.expm1(log_price)

print(f"예측 차량 가격: £{int(price):,}")
