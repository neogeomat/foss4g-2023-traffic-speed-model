# SM2T: Traffic Speed Model

Model to predict traffic speed based on OpenStreetMap, Twitter and Uber data developed within the SocialMedia2Traffic project.

## Description

The repository contains Python scripts to model traffic speed based on OpenStreetMap and Twitter data using Uber data as reference. The Twitter data is preprocessed in [socialmedia2traffic-modeller](https://github.com/GIScience/socialmedia2traffic-modeller) repo, the centrality data can be produced using the code in the [https://github.com/GIScience/socialmedia2traffic-centrality](https://github.com/GIScience/socialmedia2traffic-centrality) repo.

The traffic speed data produced within the SocialMedia2Traffic project is available through the [socialmedia2traffic-api](https://github.com/GIScience/socialmedia2traffic-api) for integration in [openrouteservice](https://openrouteservice.org/).

## Getting Started

### Dependencies

- Python >= 3.10
- PostGIS database to store data

### Installation

#### Python

After you have cloned the repository, use [poetry](https://python-poetry.org) to set up a virtual environment with all required dependencies.

```
cd sm2t_traffic_speed_model
poetry install
```

#### PostGIS Database

A postgis database is required to store the processed data. You can set up a database using docker:

```
cd sm2t_traffic_speed_model
docker compose up -d
```

To load the preprocessed input data for all cities into the database do the following:

0. Download the traffic_model_data.sql file and store it in the directory `./db-backup`.

1. Access the container’s shell:

```docker exec -it  <container_id> bash```

You get the container_id by running `docker ps`.

2. Load data into the database. This may take a while.

```psql -U postgres -W -f ./db-backup/traffic_model_backup.sql postgres```


### Executing program

The program contains three main procedures: Preprocessing, modelling and prediction.

#### 1. [preprocessing.py](preprocessing.py)

```
usage: preprocessing.py [-h] --aoi_dir AOI_DIR --config CONFIG_FILE

Preprocessing of twitter, uber and centrality data.

optional arguments:
  -h, --help            show this help message and exit
  --aoi_dir AOI_DIR, -a AOI_DIR
                        Path to the directory of the AOI
  --config CONFIG_FILE, -c CONFIG_FILE
                        Path to config file

```

##### Config file

Executing the scripts requires a config file, e.g. [./config/config_sample.json](./config/config_sample.json)

```
  "berlin": {
    "timezone": "Europe/Berlin",
    "output_dir": "/Users/chludwig/Development/sm2t/sm2t_centrality/data/extracted",
    "twitter_dir": "/Users/chludwig/Development/sm2t/sm2t_centrality/data/twitter/tweet-data-grid-timebins-weekend",
    "uber_dir": "/Users/chludwig/Development/sm2t/sm2t_centrality/data/uber"
  }
```

```
python ./traffic
```


#### 2. [modelling.py](modelling.py)

```
Train different models and compares them

optional arguments:
  -h, --help            show this help message and exit
  --aoi_dir AOI_DIR, -a AOI_DIR
                        Path to the directory of the AOI
  --config CONFIG_FILE, -c CONFIG_FILE
                        Path to config file
  --model MODEL_CONFIG_FILE, -m MODEL_CONFIG_FILE
                        Path to model config file
```


#### 3. [prediction.py](prediction.py)

```
usage: prediction.py [-h] --aoi_dir AOI_DIR --model MODEL_FILE

Predict traffic speed for OSM highways

optional arguments:
  -h, --help            show this help message and exit
  --aoi_dir AOI_DIR, -a AOI_DIR
                        Path to the directory of the AOI
  --model MODEL_FILE, - MODEL_FILE
                        Path to model file (*.joblib)
```

###

## License

This project is licensed under the [GPL3] License - see the LICENSE.md file for details

## Acknowledgments

This project was funded by the [German Federal Ministry for Digital and Transport (BMDV)](https://www.bmvi.de/EN/Home/home.html) in the context of the research initiative [mFUND](https://www.bmvi.de/EN/Topics/Digital-Matters/mFund/mFund.html).

Project term: 02/2021 - 01/2022

<p float="left">
<img src="./img/bmdv.png" height=170 align="middle" />
<img src="./img/mfund.jpg" height=170 align="middle" />
</p>

Data for model training retrieved from Uber Movement, (c) 2022 Uber Technologies, Inc., [https://movement.uber.com](https://movement.uber.com).

Twitter data retrieved from [Twitter Developer Platform](https://developer.twitter.com/en/products/twitter-api.)

OpenStreetMap data retrieved using [ohsome API](https://docs.ohsome.org/ohsome-api/v1/). (c) OpenStreetMap contributers.
