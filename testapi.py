import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {"features": [1.5, 233.2, 3.1, 44.3, 11.3]}

response = requests.post(url, json=data)
print(response.json())
