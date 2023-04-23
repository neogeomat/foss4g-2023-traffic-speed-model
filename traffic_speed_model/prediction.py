#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Predict traffic speed for all highway features"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy.types import Integer, BigInteger, Float
from dotenv import load_dotenv
from preprocessing.sql_utils import get_engine_from_environment
from utils import init_logger, FilePaths, load_config
import os
import sys

RANDOM_SEED = 22
N = 1000

# change to the working directory to the python file location so that the imports work
os.chdir(sys.path[0])

load_dotenv("../.env")

logger = init_logger("sm2t-prediction")


def add_hours(aggregated_data):
    """
    Fill missing timestamps in aggregated UBER data
    :param aggregated_data: Dataframe contain yearly
    :return:
    """
    weekday_mult = 1

    fid_arr = aggregated_data.index.get_level_values("fid").to_numpy()
    hours_arr = np.arange(0, 24)

    fid_full = np.repeat(fid_arr, weekday_mult * 24)
    hour_full = np.tile(hours_arr, weekday_mult * len(fid_arr))
    hour_df = pd.DataFrame({"fid": fid_full, "hour": hour_full}).set_index("fid")

    aggregated_data_filled = aggregated_data.join(hour_df, how="outer")
    aggregated_data_filled = aggregated_data_filled.reset_index().set_index(
        ["fid", "hour"]
    )

    return aggregated_data_filled


def predict_speed(filepaths, aoi_name: str, model_config_file: str):
    """
    Predict speed for selected pretrained models
    :param filepaths: Object containing file paths for output files
    :param model_config_file: Path to model file (.joblib) The name of the file needs to contain the features separated by "-"!
    :return:
    """
    engine = get_engine_from_environment()

    # Define parameters for models to run
    model_setups = load_config(model_config_file)
    model_evaluation_dir = filepaths.EVALUATION_DIR / Path(model_config_file).stem
    model_prediction_dir = filepaths.PREDICTION_DIR / Path(model_config_file).stem
    model_prediction_dir.mkdir(exist_ok=True)

    logger.info("Reading edges...")
    edges = pd.read_sql(
        f"SELECT fid, highway, maxspeed FROM edges_{aoi_name.replace('-', '_')}",
        engine,
    ).set_index("fid")

    logger.info("Reading centrality data...")
    centrality = pd.read_sql(
        f"SELECT fid, centrality FROM centrality_{aoi_name.replace('-', '_')}", engine
    ).set_index("fid")

    logger.info("Reading twitter by hour...")
    tweets_by_hour = pd.read_sql(
        f"SELECT * FROM twitter_by_hour_{aoi_name.replace('-', '_')}", engine
    ).set_index(["fid", "hour"])

    logger.info("Reading twitter data...")
    tweets_all = pd.read_sql(
        f"SELECT * FROM twitter_all_{aoi_name.replace('-', '_')}", engine
    ).set_index(["fid"])

    # Add hour to centrality dataframe
    centrality = add_hours(centrality)

    # Join twitter, centrality and highway tags from edges
    features = centrality.join(tweets_by_hour, how="left")
    features.fillna(0, inplace=True)
    features = features.join(tweets_all, how="left")
    features = features.join(edges, how="outer")
    features = features.reset_index().set_index("fid")

    # Standardize features, remove non numeric columns and index column
    num_cols = features.columns.drop(
        ["highway"]
    )  # exclude target variables and discrete features
    features["hour_org"] = features["hour"]

    # Standardize continuous features
    # fit the standardizer to the training features only to avoid data leakage in model
    for i in num_cols:
        scaler = StandardScaler()
        scaler.fit(features[[i]])
        features[i] = scaler.transform(features[[i]])

    # Encode nominal variables (highway tag)
    features.reset_index(inplace=True)
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(features[["highway"]])
    column_names = ohe.categories_[0].tolist()
    encoded_features = ohe.transform(features[["highway"]])

    encoded_features_df = pd.DataFrame(
        encoded_features, columns=column_names, index=features.index
    )

    features = features.join(encoded_features_df, how="outer")

    # Find models for which prediction should be performed
    for model in model_setups:
        if "predict" in model.keys() and model["predict"] == "True":
            feature_names = model["features"]
            if "highway_tag" in feature_names:
                feature_names = feature_names + column_names
                feature_names.remove("highway_tag")

            target = model["target"]
            # Load pretrained model from file and make prediction for all highways
            model_name = "-".join(model["features"])
            model_file = model_evaluation_dir / f"{model_name}.joblib"
            logger.info(f"Found model file: {model_file}")
            trained_model = joblib.load(model_file)
            logger.info(f"Predicting traffic speed...")
            feature_names_model = trained_model.feature_names_in_
            y_pred = trained_model.predict(features.loc[:, feature_names_model])

            # Save prediction to file
            pred_df = pd.DataFrame(
                {
                    "fid": features.fid,
                    target: y_pred.astype(int),
                    "hour_of_day": features.hour_org.astype(int),
                },
            )
            logger.info(f"Saving modelled traffic speed to database...")
            pred_df.to_sql(
                f"{target}_modelled_{aoi_name.replace('-', '_')}",
                engine,
                if_exists="replace",
                index=False,
                dtype={
                    "fid": BigInteger(),
                    "hour_of_day": Integer(),
                    target: Integer(),
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict traffic speed based on pretrained model"
    )
    parser.add_argument(
        "--aoi_name",
        "-a",
        required=True,
        dest="aoi_name",
        type=str,
        help="Name of the AOI to process in config file",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        dest="config_file",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        dest="model_config_file",
        type=str,
        help="Path to model config file",
    )
    args = parser.parse_args()

    aoi_name = args.aoi_name
    config_file = args.config_file
    model_config_file = args.model_config_file

    aoi_config = load_config(config_file)
    aoi_dir = Path(aoi_config["output_dir"]) / aoi_name
    filepaths = FilePaths(aoi_dir)
    filepaths.create_dirs()

    predict_speed(filepaths, aoi_name, model_config_file)
