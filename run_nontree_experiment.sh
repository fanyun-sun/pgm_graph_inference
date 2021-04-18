#!/bin/bash -ex
# Runner of nontree_approx experiment

# source setup.sh  # just in case
if [ $1 == 'make_data' ]
then
    # make test data: distributed through google drive
    # echo -e "\tCreating test dataset with MCMC labels"
    # python3 create_data.py --graph_struct path --size_range 100_100 \
    #                       --num 500 --data_mode test --mode marginal --algo mcmc \
    #                       --verbose True
    # make unlabeled training graphs: distributed through google drive
    echo -e "\tStarted generating graphs from given parameters"
    # python3 create_data.py --graph_struct barbell --size_range 100_100 \
    #                       --num 750 --data_mode train --mode marginal --algo none \
    #                       --verbose True --unlab_graphs_path barbell_train
    # python3 create_data.py --graph_struct fc --size_range 100_100 \
    #                       --num 750 --data_mode train --mode marginal --algo none \
    #                       --verbose True --unlab_graphs_path fc_train
    python3 create_data.py --graph_struct wheel --size_range 100_100 \
                          --num 750 --data_mode train --mode marginal --algo none \
                          --verbose True --unlab_graphs_path wheel_train
#     echo -e "\tNow you need to merge the two .npys with lists of graphs into nontree_train.npy"
#     python3 -c '
# import numpy as np;
# fc = np.load("./graphical_models/datasets/fc_train.npy");
# barb = np.load("./graphical_models/datasets/barbell_train.npy");
# mylist = list(barb) + list(fc);
# np.save("./graphical_models/datasets/nontrees_train.npy", mylist)
# '

elif [ $1 == 'make_labels' ]
then
    read -p 'Choose labeling algo: 1) label_prop_exact_n1_..., 2) label_sg_louvain, 3) label_tree  ' label_algo
    if [ $label_algo == 'label_prop'* ]
    then
        #make label-propagation labels for training, use format label_prop_exact_10
        echo -e "\tStarting labeling with label propagation"
    elif [ $label_algo == 'label_sg'* ]
    then
        #make label-propagation labels for training, use format label_sg_Louvain
        echo -e "\tStarting labeling with subgraph labeling"
    elif [ $label_algo == 'label_tree' ]
    then
        echo -e "\tStarting labeling with spanning tree"
    fi
    # rm -rf ./graphical_models/datasets/train/barbell  # don't want duplicating graphs
    rm -rf ./graphical_models/datasets/train/wheel
    python3 create_data.py --data_mode train --mode marginal --algo $label_algo \
                          --verbose True --unlab_graphs_path wheel_train  # used to be: nontrees_train (all at once)

elif [ $1 == 'train' ]
then
    echo -e "\tTraining your GNN"
    # pretraining stage
    python3 train.py --train_set_name trees_approx --mode marginal --epochs 5 --verbose True
    # finetuning stage
    python3 train.py --train_set_name barbell_approx --mode marginal --epochs 5 --verbose True --use_pretrained trees_approx
    # python3 train.py --train_set_name wheel_approx --mode marginal --epochs 5 --verbose True --use_pretrained trees_approx

elif [ $1 == 'test' ]
then
    echo -e "\tRunning tests"
    python3 ./experiments/run_exps.py --exp_name barbell_approx
    # python3 ./experiments/run_exps.py --exp_name fc_approx
    python3 ./experiments/run_exps.py --exp_name wheel_approx


fi
