# -------------------------------
# data_randomforrest_model.py
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# -------- 1️ Load Data --------
data = pd.read_csv("cleaned_taxi_data.csv")
data_sample = data.sample(frac=0.15, random_state=42)
print(f"Data loaded: {data_sample.shape[0]} rows, {data_sample.shape[1]} columns")

# -------- 2️ Features & Target --------
feature_cols = [
    'trip_distance',
    'trip_duration_min',
    'hour_of_day',
    'is_weekend_or_holiday'
]

X = data_sample[feature_cols]
y = data_sample['fare_amount']

# -------- 3️ Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# -------- 4️ Random Forest Training --------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    min_samples_split=10,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

print("Training Random Forest Regressor with coordinates...")
rf.fit(X_train, y_train)
print("Training completed")

# -------- 5️ Evaluation --------
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred) #0.49
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #3.39
print(f"Model Evaluation → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# -------- 6️ Feature Importance --------
importances = rf.feature_importances_
plt.figure(figsize=(8,5))
plt.barh(feature_cols, importances, color='skyblue')
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (with coordinates)")
plt.show()

# -------- 7️ Save Model --------
joblib.dump(rf, "cab_fare_rf_coords_model_ver2.pkl")
print("Model saved as cab_fare_rf_coords_model_ver2.pkl")
