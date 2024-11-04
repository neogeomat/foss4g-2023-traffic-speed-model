# Traffic Speed Modelling using OpenStreetmap and Twitter data

The repository contains the source code to model traffic speed based on OpenStreetMap and Twitter data using Uber data as reference. It was used to generate the results for the paper:

C. Ludwig, J. Psotta, A. Buch, N. Kolaxidis, S. Fendrich, M. Zia, J. Fürle, A. Rousell, A. Zipf (2023): **Traffic speed modelling to improve travel speed estimation in openrouteservice.** FOSS4G 2023, Prizren, 26 June - 2 July 2023.

## Dependencies

- Python >= 3.10, <3.12
- poetry >=1.3.2
- docker
- gunzip

## Preparation

#### 1. Set up Python environment

After you have cloned the repository, use [poetry](https://python-poetry.org) to set up a virtual environment with all required dependencies.

```
cd foss4g-2023-traffic-speed-model
poetry install
```

To activate the environment execute `poetry shell` or run all python commands with `poetry run python ...`.

#### 2. Set up PostGIS Database

A postgis database is required to store the processed data. You can set up a database using docker:

```
cd foss4g-2023-traffic-speed-model
docker compose up -d
```

#### 3. Load preprocessed data into database

1. Download the preprocessed input data from [Zenodo](https://zenodo.org/record/7857038#.ZEetIXbP0qs) and store it in the directory `./db-backup`.

2. Load the preprocessed data into the database (this may take a while)

```
cd foss4g-2023-traffic-speed-model/db-backup
gunzip < preprocessed_2023-04-25_12_29_03.gz | docker exec -i db-traffic-speed-model psql -U postgres -d postgres`
```

#### Optional: Create a backup of the database

`docker exec -t db-traffic-speed-model pg_dumpall -c -U postgres | gzip > ./preprocessed_$(date +"%Y-%m-%d_%H_%M_%S").gz`

## Running the analysis


#### 1. Preprocessing (optional)

To reproduce the results of the paper this step is not necessary if you have [loaded the preprocessed data into the database](#load-preprocessed-data). This step only needs to be performed if you are using new data which is not yet in the database.

#### 2. Train the models

Adjust the file paths in the config file [./config/config_aois_allcities](./config/config_aois_allcities). The model parameters are defined in [./config/modelling](./config/modelling). These do not have to be adjusted to reproduce the results.

**Example:** Run model training for Nairobi with target variabel 85th percentile traffic speed

`python ./modelling.py -a nairobi -c ../config/config_aois_allcities.json -m ../config/modelling/model_features_speed_kph_p85.json`

If you want to run model training for several config files at once, you can use the batch file in [./traffic_speed_model/batch](./traffic_speed_model/batch).

`./batch_modelling.sh nairobi`

#### 3. Predict traffic speed

To predict traffic speed for certain models run. Prediction will only be performed if the value `predict = True` in the modelling config file.

`python ./prediction.py -a nairobi -c ../config/config_aois_allcities.json -m ../config/modelling/model_features_speed_kph_p85.json`

If you want to run the prediction for several config files at once, you can use the batch file in [./traffic_speed_model/batch](./traffic_speed_model/batch).

`./batch_prediction.sh nairobi`

#### 4. Export traffic speed from database for openrouteservice

To export the predicted traffic speed data run:

`python ./export.py -a nairobi -t speed_kph_p85 -c ../config/config_aois_allcities.json`

If you want to run the export for several config files at once, you can use the batch file in [./traffic_speed_model/batch](./traffic_speed_model/batch).

`./batch_prediction.sh nairobi`


## License

This project is licensed under the MIT License - see the LICENSE file for details

## Authors

- Christina Ludwig
- Anna Buch


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
