#!/bin/bash
source /home/software/anaconda3/etc/profile.d/conda.sh  #

echo "$1"
conda activate carla && python ./simulation_carla.py $1
conda activate dave 



