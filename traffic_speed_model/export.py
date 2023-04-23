#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export edges and predicted speed tables to sql files"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os
import sys
from preprocessing.sql_utils import get_engine_from_environment

os.chdir(sys.path[0])

load_dotenv("../.env", verbose=True)

from utils import init_logger, load_config

logger = init_logger("sm2t-export")


def table_to_csv(city, target, outfile):
    """Create view of polygons and green beliefs"""
    sql = f"""
    SELECT
        osm_way_id, osm_start_node_id, osm_end_node_id,
        traffic_speed.*
    FROM
        "{target}_modelled_{city}" traffic_speed
        LEFT OUTER JOIN "edges_{city}" edges ON traffic_speed.fid = edges.fid
    """
    engine = get_engine_from_environment()
    data = pd.read_sql(sql, con=engine)
    data["year"] = 2020
    data.rename(columns={target: "speed_kph_p85"}, inplace=True) # needs to be renamed because this is the column name expected by ORS
    data.drop("fid", axis=1, inplace=True)
    data.to_csv(outfile, index=False)

def uber_table_to_csv(city, target, outfile):
    """Create view of polygons and green beliefs"""
    sql = f"""
    SELECT
        osm_way_id, osm_start_node_id, osm_end_node_id, speed_kph_p85, hour
    FROM
        "uber_{city}"
    """
    engine = get_engine_from_environment()
    data = pd.read_sql(sql, con=engine)
    data["year"] = 2020
    data.rename(columns={target: "speed_kph_p85", 'hour': 'hour_of_day'}, inplace=True) # needs to be renamed because this is the column name expected by ORS
    data.to_csv(outfile, index=False)



def export_for_ors(city: str, target: str, results_dir: str):
    output_dir = Path(results_dir) / city / "model" / "prediction" / target
    output_dir.mkdir(exist_ok=True)
    if target != "uber":
        outfile = output_dir / f"{target}_{city}.csv"
        table_to_csv(city, target, str(outfile))
    else:
        outfile = output_dir / f"uber_{city}.csv"
        uber_table_to_csv(city, target, str(outfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export modelled traffic speed suitable for ORS"
    )
    parser.add_argument(
        "--aoi",
        "-a",
        required=True,
        dest="city",
        type=str,
        help="Name of city",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        dest="target",
        type=str,
        help="Type of speed",
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
    aoi_config = load_config(args.config_file)

    export_for_ors(args.city, args.target, aoi_config["output_dir"])
