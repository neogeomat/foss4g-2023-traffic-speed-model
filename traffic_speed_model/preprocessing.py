#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocessing of Twitter and centrality data"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import argparse
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

os.chdir(sys.path[0])
from utils import FilePaths, init_logger, load_config
from preprocessing.centrality import preprocess_centrality
from preprocessing.twitter import preprocess_twitter
from preprocessing.edges import preprocess_edges
from preprocessing.uber import preprocess_uber

load_dotenv("../.env")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing of twitter and centrality data."
    )
    parser.add_argument(
        "--aoi_name",
        "-a",
        required=True,
        dest="aoi_name",
        type=str,
        help="Name of Area-of-Interest (AOI) in config file",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        dest="config_file",
        type=str,
        help="Path to config file",
    )
    args = parser.parse_args()

    aoi_name = args.aoi_name
    config_file = args.config_file

    # Set modules to be executed to True
    run_preprocess_edges = False
    run_preprocessing_centrality = False
    run_preprocessing_twitter = True
    run_preprocessing_uber = False

    config = load_config(config_file)

    # Create output folders
    aoi_dir = Path(config["output_dir"]) / aoi_name
    fp = FilePaths(aoi_dir)
    fp.create_dirs()

    # Set up logger
    log_file = fp.PREPROCESSING_DIR / "preprocessing.log"
    logger = init_logger("sm2t-preprocessing", log_file)

    # Start running processing modules
    if run_preprocess_edges:
        preprocess_edges(fp, aoi_name)

    if run_preprocessing_centrality:
        preprocess_centrality(fp, aoi_name)

    if run_preprocessing_twitter:
        preprocess_twitter(
            fp, config["twitter_dir"], config["aois"][aoi_name]["timezone"], aoi_name
        )

    if run_preprocessing_uber:
        preprocess_uber(aoi_name, config[aoi_name]["uber_dir"])
