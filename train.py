"""
Training of the GNNInference objects
Typical training loop, resulting in saved models
(in inference/pretrained)
Authors: kkorovin@cs.cmu.edu, markcheu@andrew.cmu.edu

# TODO: use validation set for tuning

"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from experiments.exp_helpers import get_dataset_by_name
from inference import get_algorithm
from myconstants import *

# Loss computer objects that let us save a little on creating objects----------
class CrossEntropyComputer:
    def __init__(self):
        self.computer = nn.BCELoss()

    def __call__(self, output_probs, targets):
        return self.computer(output_probs, targets)

class KLDivLossComputer:
    def __init__(self):
        self.computer = nn.KLDivLoss()
    
    def __call__(self, output_probs, targets):
        logits = torch.log(output_probs)
        return self.computer(logits, targets)

class CrossEntropyMAPComputer:
    def __init__(self):
        self.computer = nn.BCELoss()

    def __call__(self, output_probs, targets):
        return self.computer(output_probs[:, 1], targets)


def parse_train_args():
    parser = argparse.ArgumentParser()

    # critical arguments, change them
    parser.add_argument('--train_set_name', type=str,
                        help='name of training set (see experiments/exp_helpers.py)')
    parser.add_argument('--train_num', type=int, default=1000000,
                        help='')
    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to train GNN to perform')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs to train for')

    # non-critical arguments, fine with default
    # model_name can be used for different hyperparameters later
    parser.add_argument('--model_name', default='default',
                        type=str, help='model name, defaults to the train_set_name')
    parser.add_argument('--data_dir', default='./graphical_models/datasets/train',
                        type=str, help='directory to load training data from')
    parser.add_argument('--model_dir', default='./inference/pretrained',
                        type=str, help='directory to save a trained model')
    parser.add_argument('--use_pretrained', default='none',
                        type=str, help='use pretrained model')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display training statistics')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_train_args()
    print("Training a model `{}` on training dataset `{}`".format(args.model_name,
                                                                  args.train_set_name))

    dataset = get_dataset_by_name(args.train_set_name, args.data_dir, args.train_num, mode=args.mode)
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, '-'.join([args.model_name, args.train_set_name, str(args.train_num)]))

    # # filter by mode:
    # if args.mode =="marginal":
    #     dataset = [g for g in dataset if g.marginal is not None]
    # elif args.mode == "map":
    #     dataset = [g for g in dataset if g.map is not None]

    # GGNN parmeters --> move to myconstants.py
    # n_hidden_states = 5
    # message_dim_P = 6
    # hidden_unit_message_dim = 64 
    # hidden_unit_readout_dim = 64
    # T = 10
    # learning_rate = 1e-2

    # number of epochs
    epochs = args.epochs

    if args.model_name == 'default':
        gnn_constructor = get_algorithm("gnn_inference")
    else:
        gnn_constructor = get_algorithm(args.model_name)
    gnn_inference = gnn_constructor(args.mode, n_hidden_states, message_dim_P,
                                    hidden_unit_message_dim, hidden_unit_readout_dim,
                                    T, sparse=USE_SPARSE_GNN)
    if args.use_pretrained != 'none':
        model_path_pre = os.path.join(args.model_dir, args.use_pretrained)
        gnn_inference.load_model(model_path_pre)
        print(f"Model loaded from {model_path_pre}")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters', count_parameters(gnn_inference.model))
    optimizer = Adam(gnn_inference.model.parameters(), lr=learning_rate)

    if args.mode == "marginal":
        # criterion = KLDivLossComputer()
        criterion = CrossEntropyComputer()
    else:
        criterion = CrossEntropyMAPComputer()

    gnn_inference.save_model(model_path)
    print("Model saved in {}".format(model_path))

    best_epoch = -1
    best_loss = 1e9
    for epoch in range(args.epochs):
        gnn_inference.train(dataset, optimizer, criterion, DEVICE)
        loss = gnn_inference.history["loss"][-1]

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            gnn_inference.save_model(model_path)
            print('Epoch {}: loss {}'.format(epoch, loss))
            print("Model saved in {}".format(model_path))

        if epoch - best_epoch > 4:
            break
