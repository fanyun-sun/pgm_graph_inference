"""
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

        self.Q = nn.Linear(4, self.message_dim*self.message_dim)
        self.propagator1 = nn.GRUCell(self.message_dim, self.state_dim)
        self.propagator2 = nn.GRUCell(self.message_dim, self.state_dim)
        self.message_passing1 = nn.Sequential(
            nn.Linear(2*self.state_dim, self.hidden_unit_message_dim),
            # 2 for each hidden state, 1 for J[i,j], 1 for b[i] and 1 for b[j]
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        self.message_passing2 = nn.Sequential(
            nn.Linear(2*self.state_dim, self.hidden_unit_message_dim),
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
        device = J.device
        n_var_nodes = len(J)
        row, col = torch.nonzero(J).t()
        n_edges = row.shape[0]
        factors = [(row[i].item(), col[i].item()) for i in range(len(row)) if row[i] < col[i]]
        n_factors = len(factors)
        assert n_factors == n_edges//2

        var_hidden_states = torch.zeros(n_var_nodes, self.state_dim).to(J.device)
        factor_hidden_states = torch.zeros(n_factors, self.state_dim).to(J.device)

        var2fac_edge_feat = torch.zeros(n_var_nodes, n_factors, 4)
        var2fac_edge_index = []
        for i in range(len(factors)):
            # factor i is connected to node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            var2fac_edge_index.append([u, i])
            var2fac_edge_feat[u, i, :] = torch.FloatTensor([b[u].item(), b[v].item(), J[u,v].item(), J[v,u].item()])

            var2fac_edge_index.append([v, i])
            var2fac_edge_feat[v, i, :] = torch.FloatTensor([b[v].item(), b[u].item(), J[u,v].item(), J[v,u].item()])


        fac2var_edge_feat = torch.zeros(n_factors, n_var_nodes, 4)
        fac2var_edge_index = []
        for i in range(len(factors)):
            # factor i is connected to node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            fac2var_edge_index.append([i, u])
            fac2var_edge_feat[i, u, :] = torch.FloatTensor([b[u].item(), b[v].item(), J[u,v].item(), J[v,u].item()])
            fac2var_edge_index.append([i, v])
            fac2var_edge_feat[i, v, :] = torch.FloatTensor([b[v].item(), b[u].item(), J[u,v].item(), J[v,u].item()])

        var2fac_edge_index = torch.LongTensor(var2fac_edge_index).t().to(device)
        var2fac_edge_feat = var2fac_edge_feat.to(device)
        fac2var_edge_index = torch.LongTensor(fac2var_edge_index).t().to(device)
        fac2var_edge_feat = fac2var_edge_feat.to(device)

        for step in range(self.n_steps):

            row, col = var2fac_edge_index
            # shape: (#edges, self.state_dim*2)
            edge_messages = torch.cat([var_hidden_states[row,:],
                                       factor_hidden_states[col,:]], dim=-1)
            # shape: (# edges, self.message_dim)
            edge_messages = self.message_passing1(edge_messages)

            q = self.Q(var2fac_edge_feat[row, col, :])
            q = q.reshape(-1, self.message_dim, self.message_dim)
            # shape: (# edges, self.message_dim*self.feat
            # print(q.shape, edge_messages.shape)
            edge_messages = torch.matmul(q, edge_messages.unsqueeze(-1)).squeeze(-1)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            factor_hidden_states = self.propagator1(node_messages, factor_hidden_states)

            ##################################################
            ##################################################
            ##################################################

            row, col = fac2var_edge_index
            # shape: (#edges, self.state_dim*2)
            edge_messages = torch.cat([factor_hidden_states[row,:],
                                       var_hidden_states[col,:]], dim=-1)
            # shape: (# edges, self.message_dim)
            edge_messages = self.message_passing2(edge_messages)

            q = self.Q(fac2var_edge_feat[row, col, :])
            q = q.reshape(-1, self.message_dim, self.message_dim)
            # shape: (# edges, self.message_dim*self.feat
            # print(q.shape, edge_messages.shape)
            edge_messages = torch.matmul(q, edge_messages.unsqueeze(-1)).squeeze(-1)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            var_hidden_states = self.propagator2(node_messages, var_hidden_states)

        readout = self.readout(var_hidden_states)
        readout = self.softmax(readout)
        return readout
