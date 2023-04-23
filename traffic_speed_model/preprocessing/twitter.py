#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import geopandas as gpd
import pandas as pd
from pathlib import Path
from sqlalchemy.types import Integer, BigInteger
import sqlalchemy
from shapely.geometry import box

from preprocessing.sql_utils import get_engine_from_environment
from utils import init_logger
import datetime as dt
import pytz

logger = init_logger("preprocess-twitter")


def convert_utc_to_local(twitter_tiles: pd.DataFrame, timezone_name: str):
    """
    Converts utc timestamp to local time
    :param twitter_tiles: Dataframe containing twitter tile data
    :param timezone_name: Local time zone name
    :return:
    """

    local_timezone = pytz.timezone(timezone_name)

    twitter_tiles["timestamp"] = twitter_tiles["hour"].map(
        lambda x: dt.datetime(year=2020, month=7, day=1, hour=x, tzinfo=pytz.utc)
    )
    twitter_tiles["timestamp"] = twitter_tiles["timestamp"].dt.tz_convert(
        local_timezone
    )
    twitter_tiles["hour"] = (
        twitter_tiles["timestamp"].map(lambda x: x.hour).astype("int")
    )
    return twitter_tiles


def preprocess_twitter(filepaths, twitter_dir, timezone_name, aoi_name):
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
    ).set_index("fid")

    twitter_dir = Path(twitter_dir)

    resolutions = ["50m", "100m", "500m", "1000m"]
    for i, r in enumerate(resolutions):
        logger.info(f"Processing {r} grid...")
        found = list(twitter_dir.glob(f"{aoi_name}*{r}.csv"))
        if len(found) == 0:
            logger.warning(f"No {r} file found.")
            continue
        logger.info(f"Processing {found[0]}")
        twitter_tiles = pd.read_csv(found[0])

        # Remove weekend and sum tweets for each grid cell from weekends and weekdays
        twitter_tiles.drop("weekendBool", axis=1, inplace=True)
        twitter_tiles = (
            twitter_tiles.groupby(["xMin", "yMin", "xMax", "yMax", "hour"])
            .sum()
            .reset_index()
        )

        # Convert hour to local time zone
        twitter_tiles = convert_utc_to_local(twitter_tiles, timezone_name)

        # Use 3 hour interval (was a test but did not yield better results in the model)
        # data_3H = data.groupby(['xMin', 'yMin', 'xMax', 'yMax', 'weekendBool', pd.Grouper(key='timestamp', freq='3H', label="right")]).sum()
        # data_3H.reset_index(inplace=True)
        # data_3H["hour"] = data_3H.timestamp.map(lambda x: x.hour)
        # data = data_3H

        # Create shapely geometries from bounding box coordinates
        twitter_tiles["geometry"] = twitter_tiles.apply(
            lambda x: box(x["xMin"], x["yMin"], x["xMax"], x["yMax"]), axis=1
        )
        twitter_tiles = gpd.GeoDataFrame(twitter_tiles, crs="epsg:4326")

        # Join edges and twitter tiles
        edges_twitter = gpd.sjoin(edges, twitter_tiles, how="left")

        # Tweets per hour
        tweets_per_edge_hour = (
            edges_twitter.reset_index().groupby(["fid", "hour"])["tweetSum"].mean()
        )
        tweets_per_edge_hour.rename(f"tweets_{r}_hour", inplace=True)
        tweets_per_edge_hour = pd.DataFrame(tweets_per_edge_hour)

        # Tweets per day
        tweets_per_edge_day = (
            tweets_per_edge_hour.reset_index()
            .groupby(["fid"])
            .sum()[[f"tweets_{r}_hour"]]
        )
        tweets_per_edge_day.rename(
            columns={f"tweets_{r}_hour": f"tweets_{r}_day"}, inplace=True
        )

        # Merge tweets per hour and per day
        tweet_features = tweets_per_edge_hour.join(tweets_per_edge_day, how="left")
        tweet_features = tweet_features.reset_index().set_index(["fid", "hour"])

        if i == 0:
            all_proxies = tweet_features
        else:
            all_proxies = all_proxies.join(tweet_features, how="outer")

    # Twitter city count
    twitter_total_highway = twitter_tiles.groupby("hour").sum()["tweetSum"]
    twitter_total_highway.name = "tweets_city"
    all_proxies = all_proxies.reset_index().join(
        twitter_total_highway, on="hour", how="left"
    )

    all_proxies.fillna(0, inplace=True)

    navalues = all_proxies.isna().sum(axis=0)
    if any(navalues > 0) > 0:
        logger.warning("There are still NA values in twitter features.")

    logger.info("Writing to file...")
    all_proxies.to_sql(
        f"twitter_{aoi_name.replace('-', '_')}",
        engine,
        if_exists="replace",
        index=True,
        chunksize=50000,
        dtype={
            "fid": BigInteger(),
            "hour": Integer(),
            "tweets_50m_hour": Integer(),
            "tweets_50m_day": Integer(),
            "tweets_100m_hour": Integer(),
            "tweets_100m_day": Integer(),
            "tweets_500m_hour": Integer(),
            "tweets_500m_day": Integer(),
            "tweets_1000m_hour": Integer(),
            "tweets_1000m_day": Integer(),
            "tweets_city": BigInteger(),
        },
    )


if __name__ == "__main__":
    aoi_dir = "/Users/chludwig/Development/sm2t/sm2t_centrality/data/extracted/seattle"
    twitter_dir = "/Users/chludwig/Development/sm2t/sm2t_centrality/data/twitter/tweet-data-grid-timebins-weekend"
    timezone_name = "US/Pacific"

    """
    parser = argparse.ArgumentParser(
        description="Preprocess uber data"
    )
    parser.add_argument(
        "--aoi_dir",
        "-a",
        required=True,
        dest="aoi_dir",
        type=str,
        help="AOI directory",
    )
    parser.add_argument(
        "--twitter_dir",
        "-t",
        required=True,
        dest="twitter_dir",
        type=str,
        help="Directory with twitter data",
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        dest="name",
        type=str,
        help="Name of twitter proxy",
    )

    args = parser.parse_args()
    twitter_dir = args.twitter_dir
    aoi_dir = args.aoi_dir
    name = args.name
    """
    preprocess_twitter(aoi_dir, twitter_dir, timezone_name=timezone_name)
