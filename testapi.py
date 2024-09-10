import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {"features": [1000.5, 233.2, 365487.1, 445672.3, 1765478361.3]}

response = requests.post(url, json=data)
print(response.json())
