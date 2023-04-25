#!/bin/bash

city=$1
python ../preprocessing.py -a $city -c /Users/cludwig/Development/sm2t/sm2t-traffic-speed-model/config/preprocessing/config_allcities.json
