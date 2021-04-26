"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: lingxiao@cmu.edu
         kkorovin@cs.cmu.edu
"""

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from inference.core import Inference


class MyBeliefPropagation(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    Exact BP in tree structure only need two passes,
    LBP need multiple passes until convergene. 
    """

    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def _safe_divide(self, a, b):
        '''
        Divies a by b, then turns nans and infs into 0, so all division by 0
        becomes 0.
        '''
        c = a / b
        c[c == np.inf] = 0.0
        c = np.nan_to_num(c)
        return c

    def run_one(self, graph, use_log=True, smooth=0):
        # Asynchronous BP  
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        # TODO: check more convergence conditions, like calibration
        if self.mode == "marginal": # not using log
            sumOp = logsumexp if use_log else np.sum
        else:
            sumOp = np.max

        n_nodes = graph.W.shape[0]
        g = graph.factor_graph
        g.compute_marginals(max_iter=200, tolerance=1e-20)
        results = probs = np.array([g.nodes['{}'.format(i)].marginal() for i in range(n_nodes)])
        # probs should be `
        # normalize

        # if self.mode == 'marginal':
            # if use_log:
                # results = self._safe_norm_exp(probs)
            # else:
                # results = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))

        if self.mode == 'map':
            results = np.argmax(probs, axis=1)
            results[results==0] = -1

        return results


    def run(self, graphs, use_log=True, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in tqdm(graph_iterator):
            res.append(self.run_one(graph, use_log=use_log))
        return res


if __name__ == "__main__":
    bp = BeliefPropagation("marginal")
