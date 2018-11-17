import json
import logging
import os

import pandas as pd
from flask import Flask, request
import math
import numpy as np
from clients.ml_engine import MLEngineClient

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
mle_model_name = os.getenv("GCP_MLE_MODEL_NAME")
mle_model_version = os.getenv("GCP_MLE_MODEL_VERSION")

print(project_id)

ml_engine_client = MLEngineClient(project_id, mle_model_name, mle_model_version)

def haversine_distance(origin, destination):
    """
    # Formula to calculate the spherical distance between 2 coordinates, with each specified as a (lat, lng) tuple

    :param origin: (lat, lng)
    :type origin: tuple
    :param destination: (lat, lng)
    :type destination: tuple
    :return: haversine distance
    :rtype: float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


hotspots = [[40.7669, 40.7969, -73.8700, -73.8540], # LGA
            [40.6700, 40.7100, -74.1945, -74.1545], # EWR
            [40.6150, 40.6650, -73.8323, -73.7381]] # JFK 

jfk = (40.6413, -73.7781)
ewr = (40.6895, -74.1745)
lga = (40.7769, -73.8740)

def is_hotspot(lat, lon):
    for i, hotspot in enumerate(hotspots):
        if (lat >= hotspot[0] and lat <= hotspot[1] and lon >= hotspot[2] and lon <= hotspot[3]):
            return i
    return -1


def process_test_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    raw_df['origin'] = raw_df[['pickup_latitude', 'pickup_longitude']].apply(tuple, axis=1)
    raw_df['destination'] = raw_df[['dropoff_latitude', 'dropoff_longitude']].apply(tuple, axis=1)
    raw_df['distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], x['destination']), axis=1)
    raw_df['raw_datetime'] = pd.to_datetime(raw_df['pickup_datetime'])
    raw_df['hour'] = raw_df['raw_datetime'].apply(lambda x: x.hour)
    raw_df['year'] = raw_df['raw_datetime'].apply(lambda x: x.year)

    raw_df['pickup_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['pickup_latitude'], x['pickup_longitude']), axis=1)
    raw_df['dropoff_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
    
    raw_df['pickup_JFK_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], jfk), axis=1)
    raw_df['dropoff_JFK_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], jfk), axis=1)
    raw_df['pickup_EWR_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], ewr), axis=1)
    raw_df['dropoff_EWR_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], ewr), axis=1)
    raw_df['pickup_LGA_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], lga), axis=1)
    raw_df['dropoff_LGA_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], lga), axis=1)
    test_df = raw_df[['distance', 'hour', 'year', 
                      'pickup_latitude', 'pickup_longitude',
                      'dropoff_latitude', 'dropoff_longitude',  
                      'pickup_is_hotspot', 'dropoff_is_hotspot',
                      'pickup_JFK_distance', 'dropoff_JFK_distance',
                      'pickup_EWR_distance', 'dropoff_EWR_distance',
                      'pickup_LGA_distance', 'dropoff_LGA_distance',
                      'passenger_count']]
    return test_df

@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                            convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    return json.dumps(ml_engine_client.predict(predictors_df.values.tolist()))

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
