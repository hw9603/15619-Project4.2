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
SAMPLE_PROB = 1.0  # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results
# TODO: update the value using the ID of the GS bucket, WITHOUT "gs://"
# for example, if the GS path of the bucket is gs://my-bucket
# the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'ml-fare-prediction-222320-p4ml'
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


def process_train_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    raw_df['origin'] = raw_df[['pickup_latitude', 'pickup_longitude']].apply(tuple, axis=1)
    raw_df['destination'] = raw_df[['dropoff_latitude', 'dropoff_longitude']].apply(tuple, axis=1)
    raw_df['distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], x['destination']), axis=1)
    raw_df['raw_datetime'] = pd.to_datetime(raw_df['pickup_datetime'])
    raw_df['hour'] = raw_df['raw_datetime'].apply(lambda x: x.hour)
    raw_df['year'] = raw_df['raw_datetime'].apply(lambda x: x.year)
    
    raw_df = raw_df.loc[(raw_df['pickup_latitude'] < 45) & (raw_df['pickup_latitude'] > 35)
                        & (raw_df['pickup_longitude'] < -70) & (raw_df['pickup_longitude'] > -75)
                        & (raw_df['fare_amount'] < 300) & (raw_df['passenger_count'] < 9)]
    raw_df['pickup_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['pickup_latitude'], x['pickup_longitude']), axis=1)
    raw_df['dropoff_is_hotspot'] = raw_df.apply(lambda x: is_hotspot(x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
    raw_df['pickup_JFK_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], jfk), axis=1)
    raw_df['dropoff_JFK_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], jfk), axis=1)
    raw_df['pickup_EWR_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], ewr), axis=1)
    raw_df['dropoff_EWR_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], ewr), axis=1)
    raw_df['pickup_LGA_distance'] = raw_df.apply(lambda x: haversine_distance(x['origin'], lga), axis=1)
    raw_df['dropoff_LGA_distance'] = raw_df.apply(lambda x: haversine_distance(x['destination'], lga), axis=1)
    
    train_df = raw_df[['key', 'pickup_datetime', 'distance', 'hour', 'year', 
                       'pickup_latitude', 'pickup_longitude', 
                       'dropoff_latitude', 'dropoff_longitude',
                       'pickup_is_hotspot', 'dropoff_is_hotspot',
                       'pickup_JFK_distance', 'dropoff_JFK_distance',
                       'pickup_EWR_distance', 'dropoff_EWR_distance',
                       'pickup_LGA_distance', 'dropoff_LGA_distance',
                       'fare_amount', 'passenger_count']]
    return train_df


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
    test_df = raw_df[['key', 'distance', 'hour', 'year', 
                      'pickup_latitude', 'pickup_longitude',
                      'dropoff_latitude', 'dropoff_longitude',  
                      'pickup_is_hotspot', 'dropoff_is_hotspot',
                      'pickup_JFK_distance', 'dropoff_JFK_distance',
                      'pickup_EWR_distance', 'dropoff_EWR_distance',
                      'pickup_LGA_distance', 'dropoff_LGA_distance',
                      'passenger_count']]
    return test_df


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
        default=12,
        type=int
    )

    # TODO: Create more arguments here, similar to the "max_depth" example
    # parser.add_argument(
    #     '--param2',
    #     default=...,
    #     type=...
    # )
    parser.add_argument(
        '--learning_rate',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--subsample',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--n_estimators',
        default=500,
        type=int
    )
    parser.add_argument(
        '--min_split_loss',
        default=2,
        type=int
    )
    parser.add_argument(
        '--min_child_weight',
        default=2,
        type=int
    )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        # TODO: Add the new parameters to this params dict, e.g.,
        # 'param2': args.param2
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'n_estimators': args.n_estimators,
        'min_split_loss': args.min_split_loss,
        'min_child_weight': args.min_child_weight,
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
