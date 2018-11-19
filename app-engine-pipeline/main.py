import json
import logging
import os
import base64
import wave
import requests
import sys

import pandas as pd
from flask import Flask, request, jsonify
import math
import numpy as np
from clients.ml_engine import MLEngineClient

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import texttospeech
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from googlemaps import convert
import googlemaps
from google.cloud import vision
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
mle_model_name = os.getenv("GCP_MLE_MODEL_NAME")
mle_model_version = os.getenv("GCP_MLE_MODEL_VERSION")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
automl_model_id = os.getenv("AUTOML_MODEL_ID")


ml_engine_client = MLEngineClient(project_id, mle_model_name, mle_model_version)

label_mapping = {"jing_fong": "Jing Fong",
                 "bamonte": "Bamonte",
                 "katz_deli": "Katz's Delicatessen",
                 "acme": "ACME"}

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


# lat and lng ranges for three airports in NYC
# in the format of [lat_min, lat_max, lng_min, lng_max]
hotspots = [[40.7669, 40.7969, -73.8700, -73.8540], # LGA
            [40.6700, 40.7100, -74.1945, -74.1545], # EWR
            [40.6150, 40.6650, -73.8323, -73.7381]] # JFK

jfk = (40.6413, -73.7781)
ewr = (40.6895, -74.1745)
lga = (40.7769, -73.8740)

def is_hotspot(lat, lon):
    """
    :param lat: a float for latitude
    :param lon: a float for longitude
    :return: [-1|0|1|2] denoting [none|LGA|EWR|JFK]
    """
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
    # whether it is in the range of one of the airports in NYC
    raw_df['pickup_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['pickup_latitude'], x['pickup_longitude']), axis=1)
    raw_df['dropoff_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
    # how far is the pickup or dropoff point from one of the airports
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


def speech_to_text_helper(data):
    """
    :param data: base64 encoded speech
    :return: transcript of the speech
    """
    content = base64.decodebytes(data)
    audio = speech.types.RecognitionAudio(content=content)
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)
    return response.results[0].alternatives[0].transcript

def text_to_speech_helper(text):
    """
    :param text: text for speech
    :return: generated speech for the text
    """
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.types.SynthesisInput(text=text)
    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)
    speech = str(base64.b64encode(response.audio_content).decode("utf-8"))
    return speech


def named_entities_helper(text):
    """
    :param text: a sentence string
    :return: a list of entities mentioned in the sentence
    """
    client = language.LanguageServiceClient()
    document = language.types.Document(
        content=text,
        type=language.enums.Document.Type.PLAIN_TEXT)
    entities = client.analyze_entities(document).entities
    entity_names = []
    for entity in entities:
        entity_names.append(entity.name)
    return entity_names


def directions_helper(origin, destination):
    """
    :param origin: a string for the origin name
    :param destination: a string for the destination name
    :return: latitudes and longitudes for start and end locations
    """
    client = googlemaps.Client(google_maps_api_key)

    params = {
        "origin": convert.latlng(origin),
        "destination": convert.latlng(destination)
    }
    response = client._request("/maps/api/directions/json", params).get("routes", [])[0]
    start_location = response['legs'][0]['start_location']
    end_location = response['legs'][0]['end_location']
    return start_location, end_location


def vision_helper(data):
    """
    :param origin: a base64 encoded image denoting a landmark in NYC
    :return: landmark name
    """
    content = base64.decodebytes(data.encode('utf-8'))
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=content)
    response = client.landmark_detection(image=image)
    print(response)
    labels = response.landmark_annotations
    if (len(labels) == 0):
        return ""
    return labels[0].description


def automl_vision_helper(data):
    """
    :param origin: a base64 encoded image denoting a restaurant in NYC
    :return: restaurant name
    """
    content = base64.decodebytes(data.encode('utf-8'))
    prediction_client = automl_v1beta1.PredictionServiceClient()

    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, automl_model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    print(request.payload[0])
    return label_mapping[request.payload[0].display_name]


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                            convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    return json.dumps(ml_engine_client.predict(predictors_df.values.tolist()))


@app.route('/farePredictionVision', methods=['POST'])
def farePredictionVision():
    # call API modules
    source_img = request.form['source']
    destination_img = request.form['destination']
    source_name = vision_helper(source_img)
    if (source_name == ""):
        source_name = automl_vision_helper(source_img)
    destination_name = vision_helper(destination_img)
    if (destination_name == ""):
        destination_name = automl_vision_helper(destination_img)
    start_loc, end_loc = directions_helper(source_name, destination_name)
    predict_in = {"pickup_datetime": "2018-11-18 20:05:00 UTC",
                  "pickup_longitude": start_loc['lng'],
                  "pickup_latitude": start_loc['lat'],
                  "dropoff_longitude": end_loc['lng'],
                  "dropoff_latitude": end_loc['lat'],
                  "passenger_count": 1}
    # predict
    predictors_df = process_test_data(pd.DataFrame([predict_in]))
    fare_df = ml_engine_client.predict(predictors_df.values.tolist())
    fare = fare_df[0]
    output_text = "Your expected fare from " + source_name + " to " + destination_name + " is $ " + str(round(fare, 2))
    speech = text_to_speech_helper(output_text)
    return jsonify(predicted_fare=fare,
                   entities=[source_name, destination_name],
                   text=output_text,
                   speech=speech)


@app.route('/farePrediction', methods=['POST'])
def farePrediction():
    # call API modules
    text = speech_to_text_helper(request.data)
    entity_names = named_entities_helper(text)
    start_loc, end_loc = directions_helper(entity_names[0], entity_names[1])
    predict_in = {"pickup_datetime": "2018-11-18 15:05:00 UTC",
                  "pickup_longitude": start_loc['lng'],
                  "pickup_latitude": start_loc['lat'],
                  "dropoff_longitude": end_loc['lng'],
                  "dropoff_latitude": end_loc['lat'],
                  "passenger_count": 1}
    # predict
    predictors_df = process_test_data(pd.DataFrame([predict_in]))
    fare_df = ml_engine_client.predict(predictors_df.values.tolist())
    fare = fare_df[0]
    output_text = "Your expected fare from " + entity_names[0] + " to " + entity_names[1] + " is $ " + str(round(fare, 2))
    speech = text_to_speech_helper(output_text)
    return jsonify(predicted_fare=fare,
                   entities=entity_names,
                   text=output_text,
                   speech=speech)


@app.route('/speechToText', methods=['POST'])
def speechToText():
    text = speech_to_text_helper(request.data)
    return jsonify(text=text)


@app.route('/textToSpeech', methods=['GET'])
def textToSpeech():
    text = request.args.get("text")
    speech = text_to_speech_helper(text)
    ret_map = {"speech": speech}
    return json.dumps(ret_map)


@app.route('/namedEntities', methods=['GET'])
def namedEntities():
    text = request.args.get("text")
    entity_names = named_entities_helper(text)
    ret_map = {"entities": entity_names}
    return json.dumps(ret_map)


@app.route('/directions', methods=['GET'])
def directions():
    origin = request.args.get("origin")
    destination = request.args.get("destination")
    start_location, end_location = directions_helper(origin, destination)
    ret_map = {"start_location": start_location, "end_location": end_location}
    return json.dumps(ret_map)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
