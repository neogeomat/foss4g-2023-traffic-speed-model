#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import shutil
from pathlib import Path
from sqlalchemy.types import Integer, BigInteger, Float
import datetime as dt
from preprocessing.sql_utils import get_engine_from_environment
from utils import init_logger

logger = init_logger("preprocess-uber")
MPH_to_KPH = 1.60934


def preprocess(file):
    """
    Read and preprocess UBER data
    :param file:
    :return:
    """
    file = Path(file)
    data = pd.read_csv(file)
    data["timestamp"] = pd.DatetimeIndex(data["utc_timestamp"])
    data["weekday"] = data["timestamp"].map(lambda r: 1 if r.dayofweek <= 4 else 0)
    month = int(file.stem[-1]) if file.stem[-2] == "-" else int(file.stem[-2:])
    data["month"] = month
    if "speed_mph_mean" in data.columns:
        data["speed_kph_mean"] = data["speed_mph_mean"] * MPH_to_KPH
        data["speed_kph_stddev"] = data["speed_mph_stddev"] * MPH_to_KPH
        data.drop(["speed_mph_stddev", "speed_mph_stddev"], inplace=True, axis=1)
    return data


def aggregate(data):
    """
    Aggregate UBER speed over monthy per hour and weekday
    :param data: Dataframe containing UBER data
    :return:
    """
    data_agg = data.groupby(
        ["osm_way_id", "osm_start_node_id", "osm_end_node_id", "weekday", "hour"]
    )[["timestamp"]].count()
    data_agg.columns = ["count"]
    speed_agg = data.groupby(
        [
            "osm_way_id",
            "osm_start_node_id",
            "osm_end_node_id",
            "weekday",
            "hour",
            "month",
        ]
    )[["speed_kph_mean", "speed_kph_stddev"]].mean()
    return data_agg.join(speed_agg)


def fill_missing_data(aggregated_data):
    """
    Fill missing timestamps in aggregated UBER data
    :param aggregated_data: Dataframe contain yearly
    :return:
    """
    if "weekday" in aggregated_data.index.names:
        idx_new = aggregated_data.index.droplevel([3, 4])
        weekday_mult = 2
    else:
        idx_new = aggregated_data.index.droplevel([3])
        weekday_mult = 1

    idx_new = idx_new.drop_duplicates()

    seg_id_arr = idx_new.get_level_values("osm_way_id").to_numpy()
    start_junction_id_arr = idx_new.get_level_values("osm_start_node_id").to_numpy()
    end_junction_id_arr = idx_new.get_level_values("osm_end_node_id").to_numpy()

    seg_id_full = np.repeat(seg_id_arr, weekday_mult * 24)
    start_junction_id_full = np.repeat(start_junction_id_arr, weekday_mult * 24)
    end_junction_id_full = np.repeat(end_junction_id_arr, weekday_mult * 24)
    hours_full = np.tile(np.arange(0, 24), len(seg_id_arr) * weekday_mult)
    index_levels = [
        seg_id_full,
        start_junction_id_full,
        end_junction_id_full,
        hours_full,
    ]
    index_names = ["osm_way_id", "osm_start_node_id", "osm_end_node_id", "hour"]
    if "weekday" in aggregated_data.index.names:
        weekdays_full = np.tile(
            np.concatenate([np.repeat(0, 24), np.repeat(1, 24)]), len(seg_id_arr)
        )
        index_levels.insert(3, weekdays_full)
        index_names.insert(3, "weekday")

    filled_index = pd.MultiIndex.from_arrays(index_levels, names=index_names)

    aggregated_data_filled = aggregated_data.reindex(filled_index)
    return aggregated_data_filled


def find_uber_files(uber_dir, uber_aoi_dir):
    """
    Find monthly uber files
    :param uber_dir:
    :param uber_aoi_dir:
    :return:
    """
    uber_files = []
    months = np.arange(1, 13)
    for m in months:
        found = list(uber_aoi_dir.glob(f"movement-speeds*{year}-{m}.csv"))
        if len(found) == 1:
            logger.info(f"Found {found[0]}")
            uber_files.append(found[0])
            continue
        found = list(uber_dir.glob(f"movement-speeds*{year}-{m}.csv.zip"))
        if len(found) == 1:
            logger.info(f"Found {found[0]}...")
            uber_files.append(found[0])
            continue
        else:
            logger.warning(f"No uber file found for {year}-{m}")
    return uber_files


def preprocess_uber_monthly(aoi_dir, uber_dir, year=2019):
    """
    Preprocess uber data: aggregating monthly speeds by weekday/weekend and hour
    :param uber_dir: Path to directory containing UBER data as csv files
    :param out_dir: Path to output directory
    :return:
    """
    aoi_dir = Path(aoi_dir)
    uber_dir = Path(uber_dir)
    out_dir = aoi_dir / "model"
    out_dir.mkdir(exist_ok=True)
    uber_local_dir = aoi_dir / "uber"  # uber_dir
    uber_local_dir.mkdir(exist_ok=True)

    logger.info("Searching and extracting uber files...")
    uber_files = find_uber_files(uber_dir, uber_local_dir)
    if len(uber_files) == 0:
        logger.critical("No uber files found")
        sys.exit()

    logger.info("Aggregating hourly data to monthly data...")
    datasets = []
    for f in tqdm(uber_files):
        if f.suffix == ".zip":
            logger.info("Extracting...")
            shutil.unpack_archive(f, uber_local_dir)
            unzipped = uber_local_dir / f.stem
            logger.info("Reading data...")
            data = preprocess(unzipped)
        else:
            logger.info("Reading data...")
            data = preprocess(f)
        logger.info("Aggregating...")
        data_agg = aggregate(data)
        datasets.append(data_agg)
        if f.suffix == ".zip":
            logger.info(f"Deleting {unzipped}...")
            unzipped.unlink()

    all_data = pd.concat(datasets).reset_index()

    logger.info("Aggregating monthly data to yearly data...")
    months = all_data.month.unique()
    months.sort()
    # coverages_weekday = []
    for m in tqdm(months):
        selected_data = all_data.loc[all_data["month"] <= m]
        if selected_data.empty:
            logger.warning(f"No UBER speed data before month {m}")
            continue
        aggregated_data = selected_data.groupby(
            ["osm_way_id", "osm_start_node_id", "osm_end_node_id", "weekday", "hour"]
        )[["speed_kph_mean", "speed_kph_stddev"]].quantile([0.5, 0.85])

        # aggregated_data_filled = fill_missing_data(aggregated_data)
        # coverage = (len(aggregated_data) / len(aggregated_data_filled)) * 100
        # coverages_weekday.append(coverage)

    aggregated_data.reset_index(inplace=True)
    aggregated_data = aggregated_data.astype({"weekday": "int16", "hour": "int16"})
    aggregated_data.to_csv(out_dir / f"uber_speed_{year}.csv", index=False)

    # shutil.rmtree(uber_local_dir)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(months, coverages_weekday)
    # plt.xlabel("months")
    # plt.ylabel("Coverage of highways [%]")
    # plt.title(f"Highways segments with speed info")
    # plt.savefig(out_dir / f"uber_speed_coverage.jpg", pil_kwargs={'quality': 100})


def aggregate_by_hour(uber, hour_interval=3):
    """
    Aggregates hourly speed values in given interval
    :return:
    """
    # Use 3 hour interval
    uber["timestamp"] = uber["hour"].map(
        lambda x: dt.datetime(year=2020, month=7, day=1, hour=x)
    )
    uber_3H = uber.groupby(
        [
            "osm_way_id",
            "osm_start_node_id",
            "osm_end_node_id",
            "weekday",
            pd.Grouper(key="timestamp", freq=f"{hour_interval}H", label="right"),
        ]
    ).mean()
    uber_3H.reset_index(inplace=True)
    uber_3H["hour"] = uber_3H.timestamp.map(lambda x: x.hour)
    return uber_3H


def preprocess_uber(aoi_name, uber_dir):
    """
    Preprocess uber data: aggregating monthly speeds by weekday/weekend and hour
    :param uber_dir: Path to directory containing UBER data as csv files
    :param out_dir: Path to output directory
    :return:
    """

    engine = get_engine_from_environment()

    # Find uber file for AOI
    uber_file = list(Path(uber_dir).glob(f"*{aoi_name}*.csv"))[0]
    logger.info(f"Found UBER speed file: {uber_file}")

    # Read uber data
    logger.info("Reading uber data...")
    uber = pd.read_csv(uber_file)
    uber.rename(columns={"hour_of_day": "hour"}, inplace=True)
    if "speed_mph_p50" in uber.columns:
        uber["speed_kph_mean"] = uber["speed_mph_mean"] * MPH_to_KPH
        uber["speed_kph_p50"] = uber["speed_mph_p50"] * MPH_to_KPH
        uber["speed_kph_p85"] = uber["speed_mph_p85"] * MPH_to_KPH
        uber.drop(
            ["speed_mph_p50", "speed_mph_p85", "speed_mph_mean"], inplace=True, axis=1
        )
    uber = uber[~uber["speed_kph_p85"].isna()]  # drop samples with no speed data

    uber.set_index(["osm_way_id", "osm_start_node_id", "osm_end_node_id"], inplace=True)

    # Edges
    edges = pd.read_sql(
        f"SELECT fid, osm_way_id, osm_start_node_id, osm_end_node_id FROM edges_{aoi_name.replace('-', '_')}",
        engine,
    )
    edges.set_index(
        ["osm_way_id", "osm_start_node_id", "osm_end_node_id"], inplace=True
    )

    uber = uber.join(edges[["fid"]], how="left")
    uber = uber.loc[~uber["fid"].isna()]
    uber.reset_index(inplace=True)

    logger.info("Writing uber data to database...")
    uber.to_sql(
        f"uber_{aoi_name.replace('-', '_')}",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "fid": Integer(),
            "osm_way_id": BigInteger(),
            "osm_start_node_id": BigInteger(),
            "osm_end_node_id": BigInteger(),
            "hour": Integer(),
            "speed_kph_mean": Float(),
            "speed_kph_p50": Float(),
            "speed_kph_p85": Float(),
        },
    )


if __name__ == "__main__":
    # uber_dir = "/Users/chludwig/Data/sds_hd/sd17f001/SM2T/uber-movement-data/cincinnati-usa"
    # aoi_dir = "/Users/chludwig/Development/sm2t/sm2t_centrality/data/extracted/cincinnati"
    # year = 2019

    parser = argparse.ArgumentParser(description="Preprocess uber data")
    parser.add_argument(
        "--uber_dir",
        "-u",
        required=True,
        dest="uber_dir",
        type=str,
        help="Directory with UBER speed data",
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
        "--year",
        "-y",
        required=False,
        dest="year",
        default=2019,
        type=int,
        help="Year to process",
    )
    args = parser.parse_args()
    uber_dir = args.uber_dir
    aoi_dir = args.aoi_dir
    year = args.year

    preprocess_uber_monthly(aoi_dir=aoi_dir, uber_dir=uber_dir, year=year)
