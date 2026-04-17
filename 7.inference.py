import requests
import json
import numpy as np

url = "http://localhost:8000/predict"

features = [0.0] * 30

try:
    response = requests.post(url, json={"features": features})
    print("Status Code:", response.status_code)
    print("Response Body:", response.json())
except Exception as e:
    print("Error connecting to API:", str(e))