import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load cleaned train data
train_clean = pd.read_csv("train_cleaned.csv")

# Ensure pickup_datetime is dropped
if "pickup_datetime" in train_clean.columns:
    train_clean = train_clean.drop(columns=["pickup_datetime"])

# Features and target
X = train_clean.drop(columns=["fare_amount"])
y = train_clean["fare_amount"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Validation MAE: {mae:.2f}")
print(f"Validation RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "fare_model.pkl")

