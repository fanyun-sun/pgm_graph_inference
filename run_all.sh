#!/bin/bash -ex
# Runner of small exact experiment
# run as run_small.sh make_data path


  echo -e "\tCreating train data"
  python create_data.py --graph_struct $1 --size_range 9_9 \
                        --num 1300 --data_mode train --mode marginal --algo exact \
                        --verbose True
  echo -e "\tCreating test data"
  python create_data.py --graph_struct $1 --size_range 9_9 \
                        --num 300 --data_mode test --mode marginal --algo exact \
                        --verbose True
  echo -e "\tTraining your GNN"
  python train.py --train_set_name $1_small --mode marginal --epochs 5 --verbose True

  echo -e "\tRunning tests"
  python run_exps.py --exp_name in_sample_$1

  echo -e "\tCompute MAP for bp, mcmc, and gnn"
  python compute_MAP_accuracy.py --data_file ./experiments/saved_exp_res/res_$1_small_$1_small.npy
