import pandas as pd
import holidays
from datetime import datetime

# -------------- CONFIG --------------
US_HOLIDAYS = holidays.US()
FILES = [
    "yellow_tripdata_2015-01.csv",
    "yellow_tripdata_2015-02.csv",
    "yellow_tripdata_2015-03.csv",
    "yellow_tripdata_2016-01.csv",
    "yellow_tripdata_2016-02.csv",
    "yellow_tripdata_2016-03.csv"
]
SAMPLE_FRAC = 0.10  # 10% sampling
CHUNKSIZE = 500_000  # Adjust if memory is tight
# -----------------------------------


def load_and_sample(file_path):
    print(f"Processing: {file_path}")
    chunks = pd.read_csv(file_path, chunksize=CHUNKSIZE)
    sample_chunks = [chunk.sample(frac=SAMPLE_FRAC, random_state=42) for chunk in chunks]
    df = pd.concat(sample_chunks, ignore_index=True)
    return df


def preprocess(df):
    # Keep only necessary columns
    keep_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
        'trip_distance', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'fare_amount'
    ]
    df = df[keep_cols].copy()

    # Convert datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Drop rows with missing datetimes
    df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    # Derived columns
    df['trip_duration_min'] = (
        (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    )
    df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
    df['is_weekend_or_holiday'] = df['tpep_pickup_datetime'].apply(
        lambda x: 1 if (x.weekday() >= 5 or x in US_HOLIDAYS) else 0
    )

    # Basic sanity filters
    df = df[(df['trip_distance'] > 0) & (df['trip_duration_min'] > 0)]
    df = df[df['fare_amount'] > 0]

    return df


# ----------- MAIN EXECUTION -----------
combined_dfs = []
for file in FILES:
    sampled_df = load_and_sample(file)
    cleaned_df = preprocess(sampled_df)
    combined_dfs.append(cleaned_df)

final_df = pd.concat(combined_dfs, ignore_index=True)
final_df.to_csv("cleaned_taxi_data.csv", index=False)

print("Combined cleaned dataset saved as cleaned_taxi_data.csv")
print(f"Total rows: {len(final_df)}")

