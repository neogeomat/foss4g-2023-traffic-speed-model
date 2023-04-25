#!/bin/sh
echo "Processing $1"

python ../modelling.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_p85.json
python ../modelling.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_p50.json
python ../modelling.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_features_speed_kph_mean.json
python ../modelling.py -a $1 -c ../config/config_allcities.json -m ../config/modelling/model_size_time.json
