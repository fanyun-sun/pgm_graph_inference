#!/bin/bash -ex
# Runner of small exact experiment
# run as run_small.sh make_data path

#rm graphical_models/datasets/train/$1 -rf
#rm graphical_models/datasets/test/$1 -rf

#echo -e "\tCreating train data"
#python create_data.py --graph_struct $1 --size_range 9_9 \
                      #--num 10000 --data_mode train --mode marginal --algo exact \
                      #--verbose True
#echo -e "\tCreating test data"
#python create_data.py --graph_struct $1 --size_range 9_9 \
                      #--num 300 --data_mode test --mode marginal --algo exact \
                      #--verbose True
echo -e "\tTraining your GNN"
python train.py --train_set_name $1_large --mode marginal --epochs 50 --verbose True --model_name $2

echo -e "\tRunning tests"
python run_exps.py --exp_name in_sample_$1_large --model_name $2

#echo -e "\tCompute MAP for bp, mcmc, and gnn"
#python compute_MAP_accuracy.py --data_file ./experiments/saved_exp_res/res_$1_small_$1_small_$2.npy
