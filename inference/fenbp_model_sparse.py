"""
Defines GGNN model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_scatter import scatter

class GGNN(nn.Module):
    def __init__(self, state_dim, message_dim,hidden_unit_message_dim, hidden_unit_readout_dim, n_steps=10):
        super(GGNN, self).__init__()

        self.mode = "marginal"
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.message_dim = message_dim
        self.hidden_unit_message_dim = hidden_unit_message_dim
        self.hidden_unit_readout_dim = hidden_unit_readout_dim

        # self.propagator = nn.GRUCell(self.message_dim, self.state_dim)
        # self.message_passing = nn.Sequential(
            # nn.Linear(2*self.state_dim+1+3, self.hidden_unit_message_dim),
            # # 2 for each hidden state, 1 for J[i,j], 1 for b[i] and 1 for b[j]
            # nn.ReLU(),
            # nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        # )
        self.calibration = nn.Sequential(
            nn.Linear(4, self.hidden_unit_readout_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_readout_dim, self.hidden_unit_readout_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_readout_dim, 1),
        )

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.1)
                m.weight.data.fill_(0.)
                m.bias.data.fill_(0.)

    def _safe_norm_exp(self, logit):
        tmp = torch.max(logit, dim=1, keepdims=True)
        logit = logit - tmp.values
        prob = torch.exp(logit)
        prob = prob / torch.sum(prob, dim=1, keepdims=True)
        return prob
        # logit -= np.max(logit, axis=1, keepdims=True)
        # prob = np.exp(logit)
        # prob /= prob.sum(axis=1, keepdims=True)
        # return prob

    def _safe_divide(self, a, b):
        '''
        Divies a by b, then turns nans and infs into 0, so all division by 0
        becomes 0.
        '''
        c = a / b
        c[c == np.inf] = 0.0
        c = np.nan_to_num(c)
        return c

    # unbatch version for debugging
    def forward(self, J, b):
        use_log = True
        max_iters = 10
        # epsilon = 1e-20 # determines when to stop

        row, col = np.where(J)
        n_V, n_E = len(b), len(row)
        # create index dict
        degrees = np.sum(J.numpy() != 0, axis=0)
        index_bases = np.zeros(n_V, dtype=np.int64)
        for i in range(1, n_V):
            index_bases[i] = index_bases[i-1] + degrees[i-1]

        neighbors = {i:[] for i in range(n_V)}
        for i,j in zip(row,col): neighbors[i].append(j)
        neighbors = {k: sorted(v) for k, v in neighbors.items()}
        # sort nodes by neighbor size 
        ordered_nodes = np.argsort(degrees)

        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        messages = torch.ones([n_E, 2])
        if use_log:
            messages = torch.log(messages)  # log

        xij = torch.FloatTensor([[1,-1],[-1,1]])
        xi = torch.FloatTensor([-1, 1])
        converged = False
        for iter in range(max_iters):
            # save old message for checking convergence
            old_messages = messages.clone()
            messages = messages.detach()
            # update messages
            for i in ordered_nodes:
                # print("updating message at", i)
                neighbor = neighbors[i]
                # print(neighbor)
                Jij = J[i][neighbor] # vector
                bi = b[i]            # scalar
                # print(Jij, bi)
                local_potential = Jij.reshape(-1,1,1)*xij + bi*xi.reshape(-1,1)
                # print(local_potential)
                if not use_log:
                    local_potential = np.exp(local_potential)
                # get in messages product (log)
                in_message_prod = 0 if use_log else 1
                for j in neighbor:
                    if use_log:
                        in_message_prod += messages[index_bases[j]+neighbors[j].index(i)]
                    else:

                        in_message_prod *= messages[index_bases[j]+neighbors[j].index(i)]
                # messages_ = messages.copy()
                for k in range(degrees[i]):
                    j = neighbor[k]
                    if use_log:
                        messages[index_bases[i]+k] = in_message_prod - \
                           (messages[index_bases[j]+neighbors[j].index(i)])
                    else:
                        messages[index_bases[i]+k] = self._safe_divide(in_message_prod,
                           messages[index_bases[j]+neighbors[j].index(i)])
                # update
                messages[index_bases[i]:index_bases[i]+degrees[i]] = \
                    torch.logsumexp(messages[index_bases[i]:index_bases[i]+degrees[i]].reshape(degrees[i],2,1) + local_potential, dim=1)


            probs = torch.zeros([n_V, 2])
            for i in range(n_V):
                probs[i] = b[i]*xi
                for j in neighbors[i]:
                    if use_log:
                        probs[i] += messages[index_bases[j]+neighbors[j].index(i)]

            messages = [messages[:, 0:1], messages[:, 1:2]]
            old_messages = [old_messages[:, 0:1], old_messages[:, 1:2]]

            features = torch.zeros([n_E, 2])
            for i in ordered_nodes:
                for j in neighbors[i]:
                    features[index_bases[i] + neighbors[i].index(j),0] = probs[i,0]
                    features[index_bases[i] + neighbors[i].index(j),1] = messages[0][index_bases[j] + neighbors[j].index(i),0].detach() + \
                                                                         messages[1][index_bases[j] + neighbors[j].index(i),0].detach()

                    features[index_bases[j] + neighbors[j].index(i),0] = probs[j,0]
                    features[index_bases[j] + neighbors[j].index(i),1] =  messages[0][index_bases[i] + neighbors[i].index(j),0].detach() + \
                                                                          messages[1][index_bases[i] + neighbors[i].index(j),0].detach()

            alpha = self.sigmoid(self.calibration(torch.cat([messages[0], old_messages[0], features], dim=-1)))
            messages[0] = (1.-alpha)*messages[0] + alpha*old_messages[0]

            features = torch.zeros([n_E, 2])
            for i in ordered_nodes:
                for j in neighbors[i]:
                    features[index_bases[i] + neighbors[i].index(j),0] = probs[i,1]
                    features[index_bases[i] + neighbors[i].index(j),1] = messages[1][index_bases[j] + neighbors[j].index(i),0].detach() + \
                                                                         messages[0][index_bases[j] + neighbors[j].index(i),0].detach()

                    features[index_bases[j] + neighbors[j].index(i),0] = probs[j,1]
                    features[index_bases[j] + neighbors[j].index(i),1] = messages[1][index_bases[i] + neighbors[i].index(j),0].detach() + \
                                                                         messages[0][index_bases[i] + neighbors[i].index(j),0].detach() 

            alpha = self.sigmoid(self.calibration(torch.cat([messages[1], old_messages[1], features], dim=-1)))
            messages[1] = (1.-alpha)*messages[1] + alpha*old_messages[1]

            messages = torch.cat(messages, dim=-1)


        # calculate marginal or map
        probs = torch.zeros([n_V, 2])
        for i in range(n_V):
            probs[i] = b[i]*xi
            for j in neighbors[i]:
                if use_log:
                    probs[i] += messages[index_bases[j]+neighbors[j].index(i)]
        # normalize
        if self.mode == 'marginal':
            if use_log:
                results = self._safe_norm_exp(probs)

        return results
