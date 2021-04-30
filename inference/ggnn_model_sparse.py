"""
Defines GGNN model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

class GGNN(nn.Module):
    def __init__(self, state_dim, message_dim,hidden_unit_message_dim, hidden_unit_readout_dim, n_steps=10):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_steps = n_steps
        self.message_dim = message_dim
        self.hidden_unit_message_dim = hidden_unit_message_dim
        self.hidden_unit_readout_dim = hidden_unit_readout_dim

        self.propagator = nn.GRUCell(self.message_dim, self.state_dim)
        self.message_passing = nn.Sequential(
            nn.Linear(2*self.state_dim+1+3, self.hidden_unit_message_dim),
            # 2 for each hidden state, 1 for J[i,j], 1 for b[i] and 1 for b[j]
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_unit_readout_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_readout_dim, self.hidden_unit_readout_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_readout_dim, 2),
        )

        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.fill_(0)

    # unbatch version for debugging
    def forward(self, J, b):
        n_nodes = len(J)
        row, col = torch.nonzero(J).t()
        n_edges = row.shape[0]
        # initialize node embeddings to zeros
        hidden_states = torch.zeros(n_nodes, self.state_dim).to(J.device)

        edge_feat = torch.zeros(n_nodes, n_nodes, 4).to(J.device)
        for i in range(n_edges):
            # considering edge row[i], col[i]
            edge_feat[row[i], col[i], :] = torch.FloatTensor([b[row[i]], b[col[i]], J[row[i], col[i]], J[col[i], row[i]]])

        for step in range(self.n_steps):
            # (dim0*dim1, dim2)
            edge_messages = torch.cat([hidden_states[row,:],
                                       hidden_states[col,:],
                                       edge_feat[row, col, :]], dim=-1)

            edge_messages = self.message_passing(edge_messages)
            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            hidden_states = self.propagator(node_messages, hidden_states)

        readout = self.readout(hidden_states)
        readout = self.softmax(readout)
        return readout
