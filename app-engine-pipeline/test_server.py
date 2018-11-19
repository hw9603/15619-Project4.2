"""
A short script to test the local server.
"""
import requests
import pandas as pd
import base64
# endpoint = "http://localhost:5000"
endpoint = "https://ml-fare-prediction-222320.appspot.com"
data = pd.read_csv('input.csv').to_json(orient='records')
print(data)
print(requests.post('{}/predict'.format(endpoint), data=data).json())


# print("======TEST farePredictionVision======")
# # ORI_PATH = "/home/clouduser/ProjectMachineLearning/vision_dataset/new-york-city-statue-of-liberty.jpg"
# # DEST_PATH = "/home/clouduser/ProjectMachineLearning/vision_dataset/william-wachter-136221-unsplash-e1522110392880.jpg"
# ORI_PATH = "/home/clouduser/ProjectMachineLearning/restaurants_train_set/ACME/acme_10.jpg"
# DEST_PATH = "/home/clouduser/ProjectMachineLearning/restaurants_train_set/Bamonte/bamonte_10.jpg"

# with open(ORI_PATH, 'rb') as ff:
#     ori_data = ff.read()
# with open(DEST_PATH, 'rb') as ff:
#     dest_data = ff.read()

# ori_data = str(base64.b64encode(ori_data).decode("utf-8"))
# dest_data = str(base64.b64encode(dest_data).decode("utf-8"))
# data = {'source': ori_data, 'destination': dest_data}
# print(requests.post('{}/farePredictionVision'.format(endpoint), data=data).text)