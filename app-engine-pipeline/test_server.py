"""
A short script to test the local server.
"""
import requests
import pandas as pd
endpoint = "http://localhost:5000"
data = pd.read_csv('input.csv').to_json(orient='records')
print(requests.post('{}/predict'.format(endpoint), data=data).json())
