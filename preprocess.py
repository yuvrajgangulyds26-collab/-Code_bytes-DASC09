import pandas as pd
import numpy as np

# Haversine function
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Reusable preprocessing function
def preprocess(df, is_train=True):
    # Drop missing values
    df = df.dropna()

    # Convert datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])  # drop invalid dates

    # Extract features
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["is_weekend"] = (df["pickup_datetime"].dt.weekday >= 5).astype(int)

    # Compute distance
    df["distance_km"] = df.apply(
        lambda row: haversine(
            row["pickup_longitude"], row["pickup_latitude"],
            row["dropoff_longitude"], row["dropoff_latitude"]
        ), axis=1
    )

    # If training set, remove outliers
    if is_train:
        if "fare_amount" in df.columns:  # only filter if target exists
            df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
        df = df[df["distance_km"] < 100]

    return df

# Main script (only runs when you execute this file)
if __name__ == "__main__":
    # Define expected column names for training and test sets
    train_cols = ["fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude", 
                  "dropoff_longitude", "dropoff_latitude"]

    test_cols = ["pickup_datetime", "pickup_longitude", "pickup_latitude", 
                 "dropoff_longitude", "dropoff_latitude"]

    # Load train data (first row is header)
    train = pd.read_csv("train.csv", header=0, names=train_cols)
    train_clean = preprocess(train, is_train=True)
    train_clean.to_csv("train_cleaned.csv", index=False)

    # Load test data
    test = pd.read_csv("test.csv", header=0, names=test_cols)
    test_clean = preprocess(test, is_train=False)
    test_clean.to_csv("test_cleaned.csv", index=False)

