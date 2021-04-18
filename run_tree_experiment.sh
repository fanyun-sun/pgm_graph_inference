#!/bin/bash -ex
# Runner of tree_approx experiment

# source setup.sh  # just in case
if [ $1 == 'make_data' ]
then
    # make test data: distributed through google drive
    echo -e "\tCreating test dataset with BP labels"
    python3 create_data.py --graph_struct path --size_range 100_100 \
                          --num 500 --data_mode test --mode marginal --algo bp \
                          --verbose True
    # make unlabeled training graphs: distributed through google drive
    echo -e "\tStarted generating graphs from given parameters"
    python3 create_data.py --graph_struct random_tree --size_range 100_100 \
                          --num 1500 --data_mode train --mode marginal --algo none \
                          --verbose True --unlab_graphs_path trees_train

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
    rm -rf ./graphical_models/datasets/train/random_tree  # don't want duplicating graphs
    python3 create_data.py --graph_struct random_tree --size_range 100_100 \
                          --num 1500 --data_mode train --mode marginal --algo $label_algo \
                          --verbose True --unlab_graphs_path trees_train

elif [ $1 == 'train' ]
then
    echo -e "\tTraining your GNN"
    # python train.py --train_set_name trees_approx --mode marginal --epochs 5 --verbose True --use_pretrained trees_approx
    python3 train.py --train_set_name trees_approx --mode marginal --epochs 5 --verbose True 

elif [ $1 == 'test' ]
then
    echo -e "\tRunning tests"
    python3 run_exps.py --exp_name trees_approx

fi
