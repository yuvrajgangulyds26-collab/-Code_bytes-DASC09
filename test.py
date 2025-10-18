import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "trip_distance": 5.2,
    "trip_duration_min": 45,
    "pickup_longitude": -73.985,
    "pickup_latitude": 40.758,
    "dropoff_longitude": -73.975,
    "dropoff_latitude": 40.765,
    "hour_of_day": 14,
    "is_weekend_or_holiday": 0
}

response = requests.post(url, json=data)
print(response.json())
