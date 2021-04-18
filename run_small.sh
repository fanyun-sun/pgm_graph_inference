#!/bin/bash -ex
# Runner of small exact experiment
# run as run_small.sh make_data path

if [ $1 == 'make_data' ]
then
    echo -e "\tCreating train data"
    python create_data.py --graph_struct $2 --size_range 9_9 \
                          --num 1300 --data_mode train --mode marginal --algo exact \
                          --verbose True
    echo -e "\tCreating test data"
    python create_data.py --graph_struct $2 --size_range 9_9 \
                          --num 300 --data_mode test --mode marginal --algo exact \
                          --verbose True
elif [ $1 == 'train' ]
then
    echo -e "\tTraining your GNN"
    python train.py --train_set_name $2_small --mode marginal --epochs 5 --verbose True

elif [ $1 == 'test' ]
then
    echo -e "\tRunning tests"
    python run_exps.py --exp_name in_sample_$2

fi
