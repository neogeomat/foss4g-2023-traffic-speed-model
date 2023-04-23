#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import pandas as pd
import geopandas as gpd

from preprocessing.sql_utils import get_engine_from_environment
from utils import FilePaths, init_logger

logger = init_logger("preprocess-centrality")


def preprocess_centrality(filepaths, aoi_name):
    """
    Fill missing centrality values for OSM highway segments
    :param edges:
    :param centrality:
    :return:
    """

    engine = get_engine_from_environment()

    logger.info("Reading edges file...")
    edges = pd.read_sql(
        f"SELECT fid, highway FROM edges_{aoi_name.replace('-', '_')}", engine
    ).set_index("fid")

    logger.info("Reading centrality file...")
    centrality = pd.read_csv(filepaths.CENTRALITY_FILE, usecols=["fid", "route_count"])

    # Join IDs and highway tags from edges with centrality values
    edges_centrality = edges.join(centrality.loc[:, ["route_count"]], how="left")

    # Fill no data
    logger.info("Filling missing centrality values...")
    edges_centrality["centrality"] = edges_centrality.groupby(["highway"])[
        ["route_count"]
    ].transform(lambda x: x.fillna(x.mean()))
    edges_centrality["centrality"].fillna(0, inplace=True)

    navalues = edges_centrality["centrality"].isna().sum()
    if navalues > 0:
        logger.warning(f"There are {navalues} NA values in centrality column.")

    edges_centrality.drop("highway", axis=1).to_sql(
        f"centrality_{aoi_name.replace('-', '_')}",
        engine,
        if_exists="replace",
        index=True,
    )


if __name__ == "__main__":
    aoi_dir = "/Users/chludwig/Development/sm2t/sm2t_centrality/data/extracted/seattle"
    fp = FilePaths(aoi_dir)
    fp.create_dirs()
    preprocess_centrality(fp)
