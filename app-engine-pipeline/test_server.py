"""
A short script to test the local server.
"""
import requests
import pandas as pd
# endpoint = "http://localhost:5000"
endpoint = "https://ml-fare-prediction-222320.appspot.com"
data = pd.read_csv('input.csv').to_json(orient='records')
print(data)
print(requests.post('{}/predict'.format(endpoint), data=data).json())
