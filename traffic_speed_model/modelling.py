#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Models for traffic speed prediction"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import argparse
import math
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from dotenv import load_dotenv
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from preprocessing.sql_utils import get_engine_from_environment

import matplotlib as mpl

mpl.rc("text", usetex=True)

RANDOM_SEED = 22
N = 1000
MPH_to_KPH = 1.60934

# change to the working directory to the python file location so that the imports work
os.chdir(sys.path[0])
from utils import init_logger, load_config, FilePaths

load_dotenv("../.env")


logger = init_logger("sm2t-modelling")


class ModelRun:
    """Performs a model run including training and testing"""

    def __init__(
        self,
        model,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        """
        Initialize
        :param model: A model object e.g. RandomForestRegressor()
        :param name: Name for the model (str)
        :param X_train: Training data features
        :param y_train: Training data target variable
        :param X_test: Test data features
        :param y_test: Test data target variable
        """
        self.model = model
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.residuals = None
        self.metrics = {
            "name": self.name,
            "train": len(self.X_train),
            "test": len(self.X_test),
        }

    def run(self):
        """
        Train and test model
        :return:
        """
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        self.metrics["MAE_train"] = mean_absolute_error(self.y_train, y_pred_train)
        self.metrics["RMSE_train"] = mean_squared_error(
            self.y_train, y_pred_train, squared=False
        )
        self.metrics["R2_train"] = r2_score(self.y_train, y_pred_train)

        self.y_pred = self.model.predict(self.X_test)
        self.metrics["MAE"] = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics["RMSE"] = mean_squared_error(
            self.y_test, self.y_pred, squared=False
        )
        self.metrics["R2"] = r2_score(self.y_test, self.y_pred)

        self.calc_residuals()

    def calc_residuals(self):
        """
        Get residuals
        :return:
        """
        # Resduals
        self.residuals = pd.DataFrame(
            {
                "test": self.y_test,
                "pred": self.y_pred,
                "residuals": self.y_test - self.y_pred,
            },
            index=self.y_test.index,
        )
        return self.residuals


def plot_results(residuals, X_test, model_name, out_dir):
    """
    Generate plots of residuals and write residuals to csv file
    :param residuals:
    :param X_test:
    :param model_name:
    :param out_dir:
    :return:
    """
    # Join highway tag, non-normlaized hour and geometry to residuals
    residuals = residuals.join(X_test.loc[:, ["highway", "hour"]], how="left")
    residuals.round(2).to_csv(Path(out_dir) / f"residuals_{model_name}.csv", index=True)

    # Plot scatter plot of residuals by highway tag
    logger.info("Generating scatter plot of residuals ...")
    cols = 3
    rows = math.ceil(len(residuals.highway.unique()) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 15), constrained_layout=True, sharex=True, sharey=True
    )
    for (highway, group), ax in zip(residuals.groupby("highway"), axes.flatten()):
        group.plot(x="test", y="pred", kind="scatter", ax=ax, title=highway)
        ax.plot([0, 100], [0, 100], color="grey", linestyle="--", alpha=0.4)
        ax.set_xlabel("UBER speed [kph]")
        ax.set_ylabel("Predicted speed [kph]")
    fig.suptitle("Predicted speed vs. uber speed")
    plt.savefig(
        Path(out_dir) / f"scatter_highway_{model_name}.jpg", bbox_inches="tight"
    )
    plt.close(fig)

    # boxplot of residuals by highway tag
    logger.info("Generating box plot of residuals...")
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    sns.boxenplot(y="highway", x="residuals", data=residuals, orient="h", ax=ax)
    plt.axvline(x=-10, color="grey", linestyle="--")
    plt.axvline(x=10, color="grey", linestyle="--")
    plt.axvline(x=0, color="grey", linestyle="--")
    fig.suptitle("Residuals of predicted speed vs. uber speed by highway tag")
    plt.xlabel("Residuals [kph]")
    plt.savefig(
        Path(out_dir) / f"residuals_highway_{model_name}.jpg", bbox_inches="tight"
    )
    plt.close(fig)

    # residuals by hour
    logger.info("Generating box plot of residuals by hour ...")
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    sns.boxenplot(y="hour", x="residuals", data=residuals, orient="h", ax=ax)
    plt.axvline(x=-10, color="grey", linestyle="--")
    plt.axvline(x=10, color="grey", linestyle="--")
    plt.axvline(x=0, color="grey", linestyle="--")
    fig.suptitle("Residuals of predicted speed vs. uber speed by hour")
    plt.xlabel("Residuals [kph]")
    plt.savefig(
        Path(out_dir) / f"residuals_box_hour_{model_name}.jpg", bbox_inches="tight"
    )
    plt.close(fig)


def partial_dependecy(model, X_test, feature_names, outfile):
    """
    Creates plots for feature importance
    :param model: Model instance
    :param feature_names: List of features
    :param outfile: Path to output file
    :return:
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    g = PartialDependenceDisplay.from_estimator(
        model, X_test, feature_names, kind="both", ax=ax, n_cols=2
    )
    g.axes_[0][1].set_xlabel("Hour of day")
    g.axes_[1][0].set_xlabel("Centrality")
    g.axes_[1][1].set_xlabel("Tweets total (250m)")
    g.axes_[0][1].get_legend().remove()
    g.axes_[1][0].get_legend().remove()
    g.axes_[1][1].get_legend().remove()
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()


def run_models(filepaths, aoi_name: str, model_config_file: str):
    """
    Run model with different features
    :param filepaths: FilePath instance
    :param uber_file: Path to uber file
    :param model_config_file: Path to model config file
    :return:
    """
    engine = get_engine_from_environment()

    # Define parameters for models to run
    model_setups = load_config(model_config_file)
    model_run_dir = filepaths.EVALUATION_DIR / Path(model_config_file).stem
    model_run_dir.mkdir(exist_ok=True)

    logger.info("Reading uber data...")
    uber = pd.read_sql(
        f"SELECT fid, hour, speed_kph_mean, speed_kph_p50, speed_kph_p85 FROM uber_{aoi_name.replace('-', '_')}",
        engine,
    ).set_index(["fid", "hour"])

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

    logger.info("Reading edges...")
    edges = pd.read_sql(
        f"SELECT fid, highway, maxspeed FROM edges_{aoi_name.replace('-', '_')}",
        engine,
    ).set_index("fid")

    # Join twitter, centrality and highway tags from edges
    potential_samples = uber.join(centrality, how="left")
    potential_samples = potential_samples.join(tweets_by_hour, how="left")
    potential_samples.fillna(0, inplace=True)
    potential_samples = potential_samples.join(tweets_all, how="left")
    potential_samples = potential_samples.join(edges, how="outer")
    potential_samples = potential_samples.reset_index().set_index("fid")

    # Standardize features
    num_cols = potential_samples.columns.drop(
        ["highway", "speed_kph_p50", "speed_kph_p85", "speed_kph_mean"]
    )  # exclude target variables and discrete features

    # Split training and testing
    samples = potential_samples.groupby("highway", group_keys=False).apply(
        lambda x: x.sample(min(N, len(x)), random_state=RANDOM_SEED)
    )
    train, test = train_test_split(samples, test_size=0.3, random_state=RANDOM_SEED)

    # Standardize continuous features
    # fit the standardizer to the training features only to avoid data leakage in model
    for i in num_cols:
        scaler = StandardScaler()
        scaler.fit(train[[i]])
        train[i] = scaler.transform(train[[i]])
        test[i] = scaler.transform(test[[i]])

    # Encode nominal variables (highway tag)
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(train[["highway"]])
    column_names = ohe.categories_[0].tolist()
    encoded_train = ohe.transform(train[["highway"]])
    encoded_test = ohe.transform(test[["highway"]])
    train = train.join(
        pd.DataFrame(encoded_train, columns=column_names, index=train.index)
    )
    test = test.join(pd.DataFrame(encoded_test, columns=column_names, index=test.index))

    # Model with mean filled value
    results = {}
    for i, v in enumerate(model_setups):
        logger.info(f"Running model #{i}...")
        y_train = train.loc[:, v["target"]]
        y_test = test.loc[:, v["target"]]
        selected_features = v["features"]
        if "highway" in v["features"]:
            selected_features = selected_features + column_names
            selected_features.remove("highway")
        try:
            X_train = train.loc[:, selected_features]
        except Exception as e:
            logger.warning(e)
            logger.warning("Skipping model....")
            continue
        X_test = test.loc[:, selected_features]
        model_name = "-".join(v["features"])
        model = ModelRun(
            xgb.XGBRegressor(max_depth=5),
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        model.run()
        results[i] = model.metrics

        # Partial dependecy
        feature_names = v["features"]
        feature_names.remove("highway")
        # only create the ice figure if all features are used in the model
        if len(feature_names) == 4:
            partial_dependecy(
                model.model,
                X_test,
                feature_names,
                model_run_dir / f"ice_{model_name}.jpg",
            )
        plot_results(model.residuals, samples, model_name, model_run_dir)

        # Save model to file
        dump(model.model, model_run_dir / f"{model_name}.joblib")

    df = pd.DataFrame().from_dict(results, orient="columns").T
    df.set_index("name", inplace=True)
    df = df.astype(float).round(2)
    df.to_csv(model_run_dir / "metrics.csv")

    del engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train different models and compare them"
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

    # Find and set up folders in output directory of AOI
    aoi_dir = Path(aoi_config["output_dir"]) / aoi_name
    fp = FilePaths(aoi_dir)
    fp.create_dirs()

    run_models(fp, aoi_name, model_config_file)
