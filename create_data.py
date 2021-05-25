"""
Data creation helpers and action.

For creating data labels, one can use exact or approximate inference algorithms,
    as well as scalable alternatives such as subgraph labeling and label propagation
    (see ./labeling/ directory for details).

If variable size range is supplied, each generated graph
    has randomly chosen size in range.

@authors: kkorovin@cs.cmu.edu

TODO:
* add random seeds
"""

import os
import argparse
import numpy as np
from pprint import pprint
from time import time
import matplotlib.pyplot as plt

from graphical_models import construct_binary_mrf, BinaryMRF
from inference import get_algorithm
from labeling import LabelProp, LabelSG, LabelTree


def parse_dataset_args():
    parser = argparse.ArgumentParser()

    # crucial arguments
    parser.add_argument('--graph_struct', default="star", type=str,
                        help='type of graph structure, such as star or fc')
    parser.add_argument('--size_range', default="5_5", type=str,
                        help='range of sizes, in the form "10_20"')
    parser.add_argument('--num', default=1, type=int,
                        help='number of graphs to generate')
    # manage unlabeled/labeled data
    parser.add_argument('--unlab_graphs_path', default='none',
                        type=str, help='whether to use previously created unlabeled graphs.\
                            If `none`, creates new graphs. \
                            If non-`none`, should be a path from base_data_dir')
    # should be used for train-test split
    parser.add_argument('--data_mode', default='train',
                        type=str, help='use train/val/test subdirectory of base_data_dir')

    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to perform')
    parser.add_argument('--algo', default='exact', type=str,
                        help='Algorithm to use for labeling. Can be exact/bp/mcmc,\
                        label_prop for label propagation, or label_sg for subgraph labeling')

    # no need to change the following arguments
    parser.add_argument('--base_data_dir', default='./graphical_models/datasets/',
                        type=str, help='directory to save a generated dataset')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display dataset statistics')
    args = parser.parse_args()
    return args


# Helpers ---------------------------------------------------------------------
def save_graphs(graphs, labels, args):
    # unlabeled data, save to its temporary address
    if args.algo == 'none':
        path = os.path.join(args.base_data_dir, args.unlab_graphs_path)
        np.save(path + '.npy', graphs, allow_pickle=True)
    # otherwise the data is prepared and should be saved
    else:
        for graph, res in zip(graphs, labels):
            if args.mode == "marginal":
                res_marginal, res_map = res, None
            else:
                res_marginal, res_map = None, res

            directory = os.path.join(args.base_data_dir, args.data_mode,
                                     graph.struct, str(graph.n_nodes))
            os.makedirs(directory, exist_ok=True)
            data = {"W": graph.W, "b": graph.b,
                    "marginal": res_marginal, "map": res_map}
                    # "factor_graph": graph.factor_graph}
            #pprint(data)

            t = "_".join(str(time()).split("."))
            path_to_graph = os.path.join(directory, t)
            np.save(path_to_graph, data)

def load_graphs(path):
    graphs = np.load(path, allow_pickle=True)
    return graphs


# Runner ----------------------------------------------------------------------
if __name__=="__main__":
    # parse arguments and dataset name
    args = parse_dataset_args()
    low, high = args.size_range.split("_")
    size_range = np.arange(int(low), int(high)+1)

    # construct graphical models
    # either new-data-generation or data labeling scenario
    if args.unlab_graphs_path == 'none' or args.algo == 'none':
        # create new graphs
        graphs = []
        for _ in range(args.num):
            # sample n_nodes from range
            n_nodes = np.random.choice(size_range)
            graphs.append(construct_binary_mrf(args.graph_struct, n_nodes))
    else:  # both are non-None: need to load data and label it
        path = os.path.join(args.base_data_dir, args.unlab_graphs_path)
        graphs = load_graphs(path + '.npy')

    # label them using a chosen algorithm
    if args.algo in ['exact', 'bp', 'mcmc']:
        algo_obj = get_algorithm(args.algo)(args.mode)
        list_of_res = algo_obj.run(graphs, verbose=args.verbose)

    # Propagate-from-subgraph algorithm (pt 2.2):
    elif args.algo.startswith('label_prop'):
        # e.g. label_prop_exact_10_5
        inf_algo_name, sg_sizes = args.algo.split('_')[2], args.algo.split('_')[3:]
        sg_sizes = list(map(int, sg_sizes))
        inf_algo = get_algorithm(inf_algo_name)(args.mode)
        label_prop = LabelProp(sg_sizes, inf_algo, max_iter=30)
        list_of_res = label_prop.run(graphs, verbose=args.verbose)

    # Subgraph labeling algorithm (pt 2.1):
    elif args.algo == 'label_tree':
        lbt = LabelTree(args.mode)
        list_of_res = lbt.run(graphs, verbose=args.verbose)

    elif args.algo.startswith('label_sg'):
        algo_method = args.algo.split('_')[2]
        # we will be using the default inf_algo
        inf_algo_name = 'exact'
        inf_algo = get_algorithm(inf_algo_name)(args.mode)
        sg_labeler = LabelSG(inf_algo, algo_method)
        list_of_res = sg_labeler.run(graphs, verbose=args.verbose)

    elif args.algo == 'none':
        list_of_res = [None] * len(graphs)
    else:
        raise ValueError(f"Labeling algorithm {args.algo} not supported.")

    # saves to final paths if labeled, otherwise to args.unlab_graphs_path
    save_graphs(graphs, list_of_res, args)
