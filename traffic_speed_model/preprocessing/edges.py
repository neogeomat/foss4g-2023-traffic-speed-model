#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import geopandas as gpd
from geoalchemy2 import Geometry
import pandas as pd
from sqlalchemy.types import Integer, BigInteger

from preprocessing.sql_utils import get_engine_from_environment
from utils import clean_maxspeed
from utils import init_logger

logger = init_logger("preprocess-edges")


def preprocess_edges(fp, aoi_name):
    """
    Preprocess edges
    :param fp:
    :return:
    """
    engine = get_engine_from_environment()
    logger.info("Reading edges ...")
    edges = gpd.read_file(fp.EDGES_FILE)
    edges = edges[
        ["fid", "u", "v", "osmid", "highway", "maxspeed", "lanes", "width", "geometry"]
    ]
    edges.rename(
        columns={
            "u": "osm_start_node_id",
            "v": "osm_end_node_id",
            "osmid": "osm_way_id",
        },
        inplace=True,
    )

    # Remove duplicates (some features have the same "osm_way_id", "osm_start_node_id", "osm_end_node_id" values
    # logger.info("Removing duplicate features ...")
    # edges = edges.drop_duplicates(
    #    subset=["osm_way_id", "osm_start_node_id", "osm_end_node_id"], keep="first"
    # )

    # Clean data
    edges["highway"] = edges["highway"].fillna("no_tag")
    # Remove non-numeric maxspeed tags and fill missing maxspeed values
    edges["maxspeed_org"] = edges["maxspeed"]
    edges["maxspeed"] = clean_maxspeed(edges["maxspeed"])
    edges["maxspeed"] = edges.groupby(["highway"])[["maxspeed"]].transform(
        lambda x: x.fillna(x.quantile(0.5))
    )
    edges["maxspeed"].fillna(edges["maxspeed"].mean(), inplace=True)
    edges["maxspeed"] = edges["maxspeed"].astype("int")

    # Todo Standardize maxspeed
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(continuous_features)
    # scaled = scaler.transform(continuous_features)

    # Refactor highway tag column to numeric value
    highway_factor, highway_idx = pd.factorize(edges.highway)
    edges["highway_tag"] = highway_factor

    edges.to_postgis(
        name=f"edges_{aoi_name.replace('-', '_')}",
        con=engine,
        if_exists="replace",
        index=False,
        dtype={
            "geometry": Geometry("LINESTRING", srid="4326"),
            "fid": BigInteger(),
            "maxspeed": Integer(),
            "highway_tag": Integer(),
        },
    )
