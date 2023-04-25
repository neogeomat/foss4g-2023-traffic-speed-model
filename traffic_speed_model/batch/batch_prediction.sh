#!/bin/sh
echo "Processing $1"

python ../prediction.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_p85.json
python ../prediction.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_p50.json
python ../prediction.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_mean.json
