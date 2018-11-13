import json
import logging
import os

import pandas as pd
from flask import Flask, request
from clients.ml_engine import MLEngineClient

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
mle_model_name = os.getenv("GCP_MLE_MODEL_NAME")
mle_model_version = os.getenv("GCP_MLE_MODEL_VERSION")

ml_engine_client = MLEngineClient(project_id, mle_model_name, mle_model_version)


def process_test_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    return raw_df

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
