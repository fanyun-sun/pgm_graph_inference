"""
Defines VGNN_sparse model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
# from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.nn.conv import GINConv, GatedGraphConv

class VGNN_sparse(nn.Module):
    def __init__(self, state_dim, message_dim,hidden_unit_message_dim, hidden_unit_readout_dim, n_steps=10):
        super(VGNN_sparse, self).__init__()

        self.state_dim = state_dim
        self.n_steps = n_steps
        self.message_dim = message_dim
        self.hidden_unit_message_dim = hidden_unit_message_dim
        self.hidden_unit_readout_dim = hidden_unit_readout_dim


        # nn1 = Sequential(Linear(self.state_dim, self.hidden_unit_message_dim),
                         # ReLU(),
                         # Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim))
        # nn2 = Sequential(Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
                         # ReLU(),
                         # Linear(self.hidden_unit_message_dim, self.state_dim))
        # self.convs = [GINConv(nn1), GINConv(nn2)]
        # self.convs = [GatedGraphConv(self.hidden_unit_message_dim, 10)]

        self.propagator = nn.GRUCell(self.message_dim, self.state_dim)
        self.message_passing = nn.Sequential(
            nn.Linear(self.state_dim*2+4, self.hidden_unit_message_dim),
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

        self.softmax = nn.Softmax(dim=1)
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.fill_(0)

    # unbatch version for debugging
    def forward(self, J, b):
        row, col = torch.nonzero(J).t()
        n_nodes = len(J)
        n_edges = row.shape[0]
        factors = [(row[i], col[i]) for i in range(len(row)) if row[i] < col[i]]
        n_factors = len(factors)
        assert n_factors == n_edges//2

        hidden_states = torch.zeros(n_nodes+n_edges//2, self.state_dim).to(J.device)
        hidden_states[:n_nodes, 0] = 0.
        hidden_states[n_nodes:, 0] = 1.

        edge_feat = torch.zeros(n_nodes+n_factors, n_nodes+n_factors, 4)
        edge_index = []
        for i in range(len(factors)):
            # consider factor idx: n_nodes+i is connected to 
            u, v = factors[i]
            # node factors[i][0], and node factors[i][1]
            edge_index.append([u, n_nodes+i])
            edge_feat[u, n_nodes+i, :] = torch.FloatTensor([0., b[u].item(), J[u,v].item(), J[v,u].item()])

            edge_index.append([v, n_nodes+i])
            edge_feat[v, n_nodes+i, :] = torch.FloatTensor([0., b[v].item(), J[u,v].item(), J[v,u].item()])
            # edge_feat[v, n_nodes+i, :] = torch.FloatTensor([0., b[v].item(), J[v,u].item(), J[u,v].item()])

            edge_index.append([n_nodes+i, u])
            edge_feat[n_nodes+i, u, :] = torch.FloatTensor([1., b[u].item(), J[u,v].item(), J[v,u].item()])

            edge_index.append([n_nodes+i, v])
            edge_feat[n_nodes+i, v, :] = torch.FloatTensor([1., b[v].item(), J[u,v].item(), J[v,u].item()])
            # edge_feat[n_nodes+i, v, :] = torch.FloatTensor([1., b[v].item(), J[v,u].item(), J[u,v].item()])

        edge_index = torch.LongTensor(edge_index).t().to(J.device)
        edge_feat = torch.FloatTensor(edge_feat).to(J.device)
        row, col = edge_index

        for step in range(self.n_steps):
            # (dim0*dim1, dim2)
            edge_messages = torch.cat([hidden_states[row, :],
                                       hidden_states[col, :],
                                       edge_feat[row, col, :]], dim=-1)


            # print(node_messages.shape)
            edge_messages = self.message_passing(edge_messages)
            # (# edges, message_dim)
            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')

            hidden_states = self.propagator(node_messages, hidden_states)

        readout = self.readout(hidden_states[:n_nodes, :])
        readout = self.softmax(readout)
        return readout
