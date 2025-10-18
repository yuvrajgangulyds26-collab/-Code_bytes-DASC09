import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load cleaned dataset or sampled version
data = pd.read_csv("cleaned_taxi_data.csv")  

# Features & target
X = data[['trip_distance', 
          'trip_duration_min',
          'is_weekend_or_holiday']]
y = data['fare_amount']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Linear Regression model...")

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Training completed.")

# Predict
y_pred = lr.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred) #6.58
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #10.26
print(f"Linear Regression -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

#graph
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Linear Regression: Predicted vs Actual Fare")
plt.show()

# Save model
joblib.dump(lr, "cab_fare_linear_regression_model.pkl")
print("Model saved as cab_fare_linear_regression_model.pkl")
