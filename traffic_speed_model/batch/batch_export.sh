#!/bin/sh
echo "Processing $1"

python ../export.py -a $1 -t speed_kph_p85 -c ../config/config_allcities.json
python ../export.py -a $1 -t speed_kph_p50 -c ../config/config_allcities.json
python ../export.py -a $1 -t speed_kph_mean -c ../config/config_allcities.json
