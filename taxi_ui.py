import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from datetime import datetime
import math

# Load trained model
model = joblib.load("fare_model.pkl")

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def predict_fare():
    try:
        pickup_long = float(entry_pickup_long.get())
        pickup_lat = float(entry_pickup_lat.get())
        dropoff_long = float(entry_dropoff_long.get())
        dropoff_lat = float(entry_dropoff_lat.get())
        
        # Get system datetime
        now = datetime.now()
        hour_of_day = now.hour
        is_weekend = 1 if now.weekday() >= 5 else 0

        # Calculate distance
        distance_km = haversine_distance(pickup_lat, pickup_long, dropoff_lat, dropoff_long)

        # Create feature vector
        features = np.array([[pickup_long, pickup_lat, dropoff_long, dropoff_lat, 
                              hour_of_day, is_weekend, distance_km]])
        
        # Predict fare
        fare = model.predict(features)[0]
        
        result_label.config(
            text=f"Predicted Fare: ${fare:.2f}\nDistance: {distance_km:.2f} km\nTime: {now}"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Tkinter UI
root = tk.Tk()
root.title("Taxi Fare Predictor")

tk.Label(root, text="Pickup Longitude:").grid(row=0, column=0, padx=5, pady=5)
entry_pickup_long = tk.Entry(root)
entry_pickup_long.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Pickup Latitude:").grid(row=1, column=0, padx=5, pady=5)
entry_pickup_lat = tk.Entry(root)
entry_pickup_lat.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Dropoff Longitude:").grid(row=2, column=0, padx=5, pady=5)
entry_dropoff_long = tk.Entry(root)
entry_dropoff_long.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Dropoff Latitude:").grid(row=3, column=0, padx=5, pady=5)
entry_dropoff_lat = tk.Entry(root)
entry_dropoff_lat.grid(row=3, column=1, padx=5, pady=5)

tk.Button(root, text="Predict Fare", command=predict_fare).grid(row=4, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.grid(row=5, column=0, columnspan=2)

root.mainloop()
