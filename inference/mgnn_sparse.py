"""
Defines GGNN model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
# from torch_geometric.nn.conv import GINConv, GatedGraphConv
# from torch.nn import Sequential, Linear, ReLU

class GGNN(nn.Module):
    def __init__(self, state_dim, message_dim,hidden_unit_message_dim, hidden_unit_readout_dim, n_steps=10):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_steps = n_steps
        self.message_dim = message_dim
        self.hidden_unit_message_dim = hidden_unit_message_dim
        self.hidden_unit_readout_dim = hidden_unit_readout_dim

        self.var2fac_propagator = nn.GRUCell(self.message_dim, self.state_dim)
        self.fac2var_propagator = nn.GRUCell(self.message_dim, self.state_dim)

        self.var2fac_message_passing = nn.Sequential(
            nn.Linear(self.state_dim*2+5, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        self.fac2var_message_passing = nn.Sequential(
            nn.Linear(self.state_dim*2+4, self.hidden_unit_message_dim),
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
        device = J.device
        row, col = torch.nonzero(J).t()
        n_edges = row.shape[0]
        factors = [(row[i].item(), col[i].item()) for i in range(len(row)) if row[i] < col[i]]
        n_factors = len(factors)
        assert n_factors == n_edges//2


        var2fac_edge_index = []
        for i in range(len(factors)):
            # factor i is connected to node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            var2fac_edge_index.append([u, i])
            var2fac_edge_index.append([v, i])
        # var2fac_edge_index = torch.LongTensor(var2fac_edge_index).t().to(J.device)
        num_var2fac_msg_nodes = len(var2fac_edge_index)
        var2fac_hidden_states = torch.zeros(num_var2fac_msg_nodes, self.state_dim).to(device)

        fac2var_edge_index = []
        for i in range(len(factors)):
            # factor i is connected to node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            fac2var_edge_index.append([i, u])
            fac2var_edge_index.append([i, v])
        num_fac2var_msg_nodes = len(fac2var_edge_index)
        fac2var_hidden_states = torch.zeros(num_fac2var_msg_nodes, self.state_dim).to(device)


        f2v_to_v2f_edge_index = []
        f2v_to_v2f_feat = torch.zeros(num_fac2var_msg_nodes, num_var2fac_msg_nodes, 5).to(device)
        v2f_to_f2v_edge_index = []
        v2f_to_f2v_feat = torch.zeros(num_var2fac_msg_nodes, num_fac2var_msg_nodes, 4).to(device)

        for ii in range(num_var2fac_msg_nodes):
            # ii is the index of var2fac_msg_node 
            # considering msg_{u -> fv}
            u, fv = var2fac_edge_index[ii]
            assert u in factors[fv]
            for jj in range(num_fac2var_msg_nodes):
                # jj is the index of fac2var msg node
                # considering msg_{fu -> v}
                fu, v = fac2var_edge_index[jj]
                assert v in factors[fu]
                if u == v and fv != fu:
                    f2v_to_v2f_edge_index.append([jj, ii])
                    tmp0 = 0
                    tmp1 = 0
                    # tmp0 = int(v == factors[fu][0])
                    # tmp1 = int(u == factors[fv][0])
                    f2v_to_v2f_feat[jj, ii, :] = torch.Tensor([b[u].item(),
                                                               J[factors[fu][tmp0], factors[fu][1-tmp0]].item(),
                                                               J[factors[fu][1-tmp0], factors[fu][tmp0]].item(),
                                                               J[factors[fv][tmp1], factors[fv][1-tmp1]].item(),
                                                               J[factors[fv][1-tmp1], factors[fv][tmp1]].item()])

                if fv == fu and u != v:
                    v2f_to_f2v_edge_index.append([ii, jj])
                    tmp0 = 0
                    # tmp0 = int(u == factors[fu][0])
                    assert factors[fu] == (u,v) or factors[fv] == (v,u)
                    v2f_to_f2v_feat[ii, jj, :] = torch.Tensor([b[u].item(),
                                                               b[v].item(),
                                                               J[factors[fu][tmp0], factors[fu][1-tmp0]].item(),
                                                               J[factors[fu][1-tmp0], factors[fu][tmp0]].item()])

        f2v_to_v2f_edge_index = torch.LongTensor(f2v_to_v2f_edge_index).t().to(device)
        v2f_to_f2v_edge_index = torch.LongTensor(v2f_to_f2v_edge_index).t().to(device)
        # var2fac_edge_feat = torch.FloatTensor(var2fac_edge_feat).to(J.device)
        # fac2var_edge_feat  = torch.FloatTensor(fac2var_edge_feat).to(J.device)

        # import ipdb;ipdb.set_trace()
        for step in range(self.n_steps):
            # f2v_to_v2f
            # calculate var2fac messages from fac2var message nodes
            # row is the indices of fac2var message nodes
            # col is the indices of var2fac message nodes
            row, col  = f2v_to_v2f_edge_index
            raw_edge_messages = torch.cat([fac2var_hidden_states[row, :],
                                           var2fac_hidden_states[col, :],
                                           f2v_to_v2f_feat[row, col, :]], dim=-1)
            edge_messages = self.var2fac_message_passing(raw_edge_messages)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            var2fac_hidden_states = self.var2fac_propagator(node_messages, var2fac_hidden_states)
            # we get the updated hidden_states of var2fac msg nodes

            # calculate fac2var messages from var2fac messages
            row, col = v2f_to_f2v_edge_index
            raw_edge_messages = torch.cat([var2fac_hidden_states[row, :],
                                           fac2var_hidden_states[col, :],
                                           v2f_to_f2v_feat[row, col, :]], dim=-1)
            edge_messages = self.fac2var_message_passing(raw_edge_messages)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            fac2var_hidden_states = self.fac2var_propagator(node_messages, fac2var_hidden_states)

        col = []
        for jj in range(num_fac2var_msg_nodes):
            # msg_{fu -> v}
            fu, v = fac2var_edge_index[jj]
            col.append(v)
        col = torch.LongTensor(col).to(device)

        # then use scatter to get node beliefes
        node_messages = scatter(fac2var_hidden_states, col, dim=0, reduce='sum')
        readout = self.readout(node_messages)
        readout = self.softmax(readout)
        return readout
