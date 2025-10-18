# -------------------------------
# data_xgboost_model.py
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

# -------- 1️ Load Data --------
data = pd.read_csv("cleaned_taxi_data.csv")
print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# -------- 2️ Features & Target --------
feature_cols = ['trip_distance','trip_duration_min','hour_of_day','is_weekend_or_holiday','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
X = data[feature_cols]
y = data['fare_amount']

# -------- 3️ Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# -------- 4️ XGBoost Model --------
xg_reg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost Regressor...")
xg_reg.fit(X_train, y_train)
print("Training completed")

# -------- 5️ Evaluation --------
y_pred = xg_reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred) #0.44
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #21.60
print(f"Model Evaluation → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# -------- 6️ Feature Importance --------
xgb.plot_importance(xg_reg, height=0.5)
plt.title("XGBoost Feature Importance")
plt.show()

# -------- 7️ Save Model --------
joblib.dump(xg_reg, "cab_fare_xgboost_model.pkl")
print("Model saved as cab_fare_xgboost_model.pkl")
