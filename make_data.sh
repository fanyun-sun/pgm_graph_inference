#!/bin/bash -ex
# Runner of small exact experiment
# run as run_small.sh make_data path

rm graphical_models/datasets/train/$1 -rf
rm graphical_models/datasets/test/$1 -rf

echo -e "\tCreating train data"
python create_data.py --graph_struct $1 --size_range 16_16 \
                      --num 5000 --data_mode train --mode marginal --algo exact \
                      --verbose True

echo -e "\tCreating test data"
python create_data.py --graph_struct $1 --size_range 16_16 \
                      --num 1000 --data_mode test --mode marginal --algo exact \
                      --verbose True
