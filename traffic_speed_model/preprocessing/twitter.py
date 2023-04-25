#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import geopandas as gpd
import pandas as pd
from pathlib import Path

import utm
from pyproj import CRS
from shapely import Point
from sqlalchemy.types import Integer, BigInteger
import sqlalchemy
from shapely.geometry import box

from preprocessing.sql_utils import get_engine_from_environment
from utils import init_logger
import datetime as dt
import pytz

logger = init_logger("preprocess-twitter")


def convert_utc_to_local(twitter_df: pd.DataFrame, timezone_name: str):
    """
    Converts utc timestamp to local time
    :param twitter_tiles: Dataframe containing twitter tile data
    :param timezone_name: Local time zone name
    :return:
    """
    local_timezone = pytz.timezone(timezone_name)
    twitter_df["timestamp"] = pd.to_datetime(
        twitter_df["createdAT"].map(lambda x: x["$date"])
    )
    twitter_df["timestamp_local"] = twitter_df["timestamp"].dt.tz_convert(
        local_timezone
    )
    twitter_df["hour"] = twitter_df["timestamp"].map(lambda x: x.hour).astype("int")
    return twitter_df


def get_utm_zone(geometry):
    """Reproject a dataframe with epsg:4326 to UTM in respective zone
    Reprojects a dataframe to UTM
    :param aoifile: Path to AOI file
    :return:
    """
    center = geometry.centroid
    utm_zone = utm.from_latlon(center.y, center.x)
    hemisphere = "south" if utm_zone[3] in "CDEFGHJKLM" else "north"
    proj4_string = f"+proj=utm +zone={utm_zone[2]} +{hemisphere}"
    crs = CRS.from_string(proj4_string)
    return crs.to_epsg()


def preprocess_twitter(twitter_dir, timezone_name, aoi_name):
    """
    Preprocess twitter data
    :return:
    """
    engine = get_engine_from_environment()

    logger.info("Reading edges...")
    edges = gpd.read_postgis(
        f"SELECT fid, geometry FROM edges_{aoi_name.replace('-', '_')}",
        engine,
        geom_col="geometry",
    )
    utmzone = get_utm_zone(edges.geometry[0])
    edges = edges.to_crs(utmzone)

    # read twitter data
    twitter_file = (
        Path(twitter_dir) / f"{aoi_name}-2018-01-01T00-00-00Z_2020-03-31T23-59-59Z.json"
    )
    twitter_df = pd.read_json(twitter_file, lines=True)
    twitter_df = twitter_df[["coords", "createdAT"]]
    twitter_df = convert_utc_to_local(twitter_df, timezone_name)
    twitter_df["geometry"] = twitter_df.coords.map(lambda x: Point(x))
    twitter_df = gpd.GeoDataFrame(twitter_df, crs="epsg:4326")
    twitter_df = twitter_df.to_crs(utmzone)
    twitter_df.drop(
        ["coords", "createdAT", "timestamp", "timestamp_local"], axis=1, inplace=True
    )
    twitter_df.hour = twitter_df.hour.astype("int")

    resolutions = [500, 50, 100, 250]
    for i, r in enumerate(resolutions):
        logger.info(f"Processing {r} grid...")

        edges_buffered = edges.copy()
        edges_buffered.geometry = edges_buffered.buffer(r)

        # Join edges and twitter tiles
        logger.info("Performing spatial join ...")
        edges_twitter = gpd.sjoin(edges_buffered, twitter_df, how="left").drop(
            ["geometry", "index_right"], axis=1
        )

        # Tweets per hour
        logger.info("Calculating tweets per hour ...")
        twitter_hour = edges_twitter.groupby(["fid", "hour"]).count()[["timestamp"]]
        twitter_hour.rename(columns={"timestamp": f"tweets_{r}_hour"}, inplace=True)

        # All Tweets
        logger.info("Calculating tweets total ...")
        twitter_all = edges_twitter.groupby(["fid"]).count()[["timestamp"]]
        twitter_all.rename(columns={"timestamp": f"tweets_{r}_all"}, inplace=True)

        logger.info("Joining data with previous results ...")
        if i == 0:
            twitter_hour_all = twitter_hour
        else:
            twitter_hour_all = twitter_hour_all.join(twitter_hour, how="outer")

        if i == 0:
            twitter_all_all = twitter_all
        else:
            twitter_all_all = twitter_all_all.join(twitter_all, how="outer")

    # Set 0 as tweet count for highway segments with NAN
    logger.info("Filling nan values ...")

    twitter_hour_all.fillna(0, inplace=True)
    twitter_all_all.fillna(0, inplace=True)

    logger.info("Writing to database...")
    twitter_hour_all.to_sql(
        f"twitter_by_hour_{aoi_name.replace('-', '_')}",
        engine,
        if_exists="replace",
        index=True,
        chunksize=50000,
        dtype={
            "fid": BigInteger(),
            "hour": Integer(),
            "tweets_50m_hour": Integer(),
            "tweets_100m_hour": Integer(),
            "tweets_500m_hour": Integer(),
            "tweets_1000m_hour": Integer(),
        },
    )

    logger.info("Writing to database...")
    twitter_all_all.to_sql(
        f"twitter_all_{aoi_name.replace('-', '_')}",
        engine,
        if_exists="replace",
        index=True,
        chunksize=50000,
        dtype={
            "fid": BigInteger(),
            "tweets_50m_all": Integer(),
            "tweets_100m_all": Integer(),
            "tweets_500m_all": Integer(),
            "tweets_1000m_all": Integer(),
        },
    )
