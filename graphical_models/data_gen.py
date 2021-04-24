"""

Graphical model generators
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
import networkx as nx
from myconstants import ASYMMETRIC

from graphical_models.data_structs import BinaryMRF

from factor_graph import Variable, Factor, FactorGraph

def to_factor_graph(graph):
    n_nodes = graph.W.shape[0]
    g = FactorGraph(silent=True)
    variables = [Variable('{}'.format(i), 2) for i in range(n_nodes)]

    number_of_factors = 0
    for i in range(n_nodes):
        factor_name =  'f{}'.format(i)
        f_ = Factor(factor_name,
                    np.array([np.exp(-graph.b[i]), np.exp(graph.b[i])]))
        g.add(f_)
        g.append(factor_name, variables[i])

        for j in range(i+1, n_nodes):
            if graph.W[i, j] != 0.:
                factor_name = 'f{}{}'.format(i,j)
                fij = Factor(factor_name, np.array([
                    [np.exp((graph.W[i,j] + graph.W[j,i])/2), np.exp(-graph.W[i,j])],
                    [np.exp(-graph.W[j, i]), np.exp((graph.W[i,j]+ graph.W[j, i])/2)]
                ]))
                g.add(fij)
                g.append(factor_name, variables[i])
                g.append(factor_name, variables[j])
                number_of_factors += 1
    return g



struct_names = ["star", "random_tree", "powerlaw_tree", "path",
                "cycle", "ladder", "grid",
                "circ_ladder", "barbell", "loll", "wheel",
                "bipart", "tripart", "fc"]

def generate_struct_mask(struct, n_nodes, shuffle_nodes):
    # a horrible collection of ifs due to args in nx constructors
    if struct == "star":
        g = nx.star_graph(n_nodes)
    elif struct == "random_tree":
        g = nx.random_tree(n_nodes)
    elif struct == "powerlaw_tree":
        g = nx.powerlaw_tree(n_nodes, gamma=3, seed=None)
    elif struct == "binary_tree":
        raise NotImplementedError("Implement a binary tree.")
    elif struct == "path":
        g = nx.path_graph(n_nodes)
    elif struct == "cycle":
        g = nx.cycle_graph(n_nodes)
    elif struct == "ladder":
        g = nx.ladder_graph(n_nodes)
    elif struct == "grid":
        n = m = int(np.sqrt(n_nodes))
        assert n*m == n_nodes
        # m = np.random.choice(range(1, n_nodes+1))
        # n = n_nodes // m
        g = nx.generators.lattice.grid_2d_graph(m, n)
    elif struct == "circ_ladder":
        g = nx.circular_ladder_graph(n_nodes)
    elif struct == "barbell":
        assert n_nodes >= 4
        m = np.random.choice(range(2, n_nodes-1))
        blocks = (m, n_nodes-m)
        g = nx.barbell_graph(*blocks)
    elif struct == "loll":
        assert n_nodes >= 2
        m = np.random.choice(range(2, n_nodes+1))
        g = nx.lollipop_graph(m, n_nodes-m)
    elif struct == "wheel":
        g = nx.wheel_graph(n_nodes)
    elif struct == "bipart":
        m = np.random.choice(range(n_nodes))
        blocks = (m, n_nodes-m)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "tripart":
        # allowed to be zero
        m, M = np.random.choice(range(n_nodes), size=2)
        if m > M:
            m, M = M, m
        blocks = (m, M-m, n_nodes-M)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "fc":
        g = nx.complete_graph(n_nodes)
    else:
        raise NotImplementedError("Structure {} not implemented yet.".format(struct))

    # fix bugs, relabel nodes to make sure nodes are indexed by integers!
    mapping = {n:idx for idx,n in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)

    node_order = list(range(n_nodes))
    if shuffle_nodes:
        np.random.shuffle(node_order)

    # a weird subclass by default; raises a deprecation warning
    # with a new update of networkx, this should be updated to
    # nx.convert_matrix.to_numpy_array
    np_arr_g = nx.to_numpy_matrix(g, nodelist=node_order)
    return np_arr_g.astype(int)


def construct_binary_mrf(struct, n_nodes, shuffle_nodes=True):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "path", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
        shuffle_nodes {bool} -- whether to permute node labelings
                                uniformly at random
    Returns:
        BinaryMRF object
    """
    W = np.random.normal(0., 1., (n_nodes, n_nodes))
    if not ASYMMETRIC:
        W = (W + W.T) / 2
    b = np.random.normal(0., 0.25**2, n_nodes)
    mask = generate_struct_mask(struct, n_nodes, shuffle_nodes)
    W *= mask
    graph = BinaryMRF(W, b, struct=struct)
    graph.factor_graph = to_factor_graph(graph)
    return graph


if __name__ == "__main__":
    print("Testing all structures:")
    n_nodes = 5
    for struct in struct_names:
        print(struct, end=": ")
        graph = construct_binary_mrf(struct, n_nodes)
        print("ok")
        # print(graph.W, graph.b)

    print("Nodes not shuffled:")
    graph = construct_binary_mrf("wheel", n_nodes, False)
    print(graph.W, graph.b)

    print("Nodes shuffled:")
    graph = construct_binary_mrf("wheel", 5)
    print(graph.W, graph.b)

    try:
        graph = construct_binary_mrf("fully_conn", 3)
    except NotImplementedError:
        pass

