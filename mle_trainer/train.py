from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os

# ==========================
# ==== Define Variables ====
# ==========================
# When dealing with a large dataset, it is practical to randomly sample
# a smaller proportion of the data to reduce the time and money cost per iteration.
#
# You should start with a low proportion, and increase it
# after you are able to better estimate how long the training time will be
# under different settings, using different proportions of the training set.
# When you are testing, start with 0.2. You need to change it to 1.0 when you make submissions.
# TODO: Set it to 1.0 when you make submissions
SAMPLE_PROB = 0.2  # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results
# TODO: update the value using the ID of the GS bucket, WITHOUT "gs://"
# for example, if the GS path of the bucket is gs://my-bucket
# the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'YOUR_OUTPUT_BUCKET'
# DO NOT change it
DATA_BUCKET_ID = 'p42ml'
# DO NOT change it
TRAIN_FILE = 'data/cc_nyc_fare_train.csv'


# =========================
# ==== Utility Methods ====
# =========================
def haversine_distance(origin, destination):
    """
    Calculate the spherical distance from coordinates

    :param origin: tuple (lat, lng)
    :param destination: tuple (lat, lng)
    :return: Distance in km
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


# =====================================
# ==== Define data transformations ====
# =====================================

def process_train_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    return raw_df


def process_test_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    return raw_df


if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================
    # You are provided with the code that creates an argparse.ArgumentParser
    # to parse the command line arguments and pass these parameters to ML Engine to
    # be tuned by HyperTune enabled.
    # TODO: Your task is to add at least 3 more arguments.
    # You need to update both the code below and config.yaml.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',  # MLE passes this in by default
        required=True
    )

    # the 5 lines of code below parse the --max_depth option from the command line
    # and will convert the value into "args.max_depth"
    # "args.max_depth" will be passed to XGBoost training through the `params` variables
    # i.e., xgb.train(params, ...)
    #
    # the 5 lines match the following YAML entry in `config.yaml`:
    # - parameterName: max_depth
    #   type: INTEGER
    #   minValue: 4
    #   maxValue: 10
    # "- parameterName: max_depth" matches "--max_depth"
    # "type: INTEGER" matches "type=int""
    # "minValue: 4" and "maxValue: 10" match "default=6"
    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )

    # TODO: Create more arguments here, similar to the "max_depth" example
    # parser.add_argument(
    #     '--param2',
    #     default=...,
    #     type=...
    # )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        # TODO: Add the new parameters to this params dict, e.g.,
        # 'param2': args.param2
    }

    """
    DO NOT CHANGE THE CODE BELOW
    """
    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
