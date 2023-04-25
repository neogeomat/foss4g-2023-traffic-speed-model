#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

from pathlib import Path
from typing import Union

import pandas as pd
import logging
import os
import json


MPH_to_KPH = 1.60934


def load_config(config_file):
    """
    Load config file
    :param config_file: path to config file (str)
    :return:
    """
    assert os.path.exists(
        config_file
    ), f"Configuration file does not exist: {os.path.abspath(config_file)}"
    with open(config_file, "r") as src:
        config = json.load(src)
    return config


def init_logger(name, log_file=None):
    """
    Set up a logger instance with stream and file logger
    :param name: Name of logger (str)
    :param log_file: path to log file (str)
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d-%Y %I:%M:%S",
    )
    # Add stream handler
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    # Log file handler
    if log_file:
        assert os.path.exists(
            os.path.dirname(log_file)
        ), "Error during logger setup: Directory of log file does not exist."
        filehandler = logging.FileHandler(filename=log_file)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    return logger


class FilePaths:
    """Managemes folder structure of output data"""

    def __init__(self, aoi_dir: Union[str, Path]):
        """Init"""

        self.AOI_DIR = Path(aoi_dir) if not isinstance(aoi_dir, Path) else aoi_dir

        self.MODEL_DIR = self.AOI_DIR / "model"

        self.PREPROCESSING_DIR = self.MODEL_DIR / "preprocessing"
        self.TWITTER_PREPROC_FILE = self.PREPROCESSING_DIR / "twitter_preprocessed.csv"
        self.UBER_PREPROC_FILE = self.PREPROCESSING_DIR / "uber_preprocessed.csv"
        self.CENTRALITY_PREPROC_FILE = (
            self.PREPROCESSING_DIR / "centrality_preprocessed.csv"
        )

        self.CENTRALITY_DIR = self.AOI_DIR / "centrality"
        self.EDGES_FILE = self.CENTRALITY_DIR / "temp" / "network" / "edges.shp"
        self.CENTRALITY_FILE = self.CENTRALITY_DIR / "centrality.csv"

        self.EVALUATION_DIR = self.MODEL_DIR / "evaluation"

        self.PREDICTION_DIR = self.MODEL_DIR / "prediction"
        self.MODELLED_SPEED_FILE = self.PREDICTION_DIR / "modelled_traffic_speed.csv"

    def create_dirs(self):
        """
        Creats sub directories
        :return:
        """
        for var, path in self.__dict__.items():
            if str(var).endswith("_DIR"):
                path.mkdir(parents=True, exist_ok=True)


def clean_maxspeed(maxspeed: pd.Series):
    """
    Cleans the maxspeed column to have only numberic values. Somtimes it also contains strings like "DE:urban", "5 mph" or none.
    All of them will be replaced by np.nan.
    :return:
    """
    return maxspeed.map(lambda x: check_types(x))


def check_types(x):
    """
    Converts strings to floats, the ones that cannot be converted will be returned as None
    :param x: Variable to be converted
    :return:
    """

    if isinstance(x, str):
        if x.isnumeric():
            return float(x)
    else:
        return x
