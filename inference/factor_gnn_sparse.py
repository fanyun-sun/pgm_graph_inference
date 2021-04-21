"""
Defines GGNN model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.conv import GINConv, GatedGraphConv
from torch.nn import Sequential, Linear, ReLU


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropagation layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class Special3dSpmm(nn.Module):
    # sparse matrix a is (n_dim0, n_dim1, n_dim2)
    # full matrix b is (n_dim2, n_dim3)
    # perform a.matmul(b), output shape is (n_dim1, n_dim3, n_dim0)
    # if n_dim3 ==1, output shape is (n_dim1, n_dim0)
    def forward(self, indices, values, shape, b):
        idx0, idx1, idx2 = indices 
        # all have size #msg * msg_dim
        n_dim0, n_dim1, n_dim2 = shape
        # msg_dim, n_var_nodes, n_var_nodes
        out = []
        for i in range(n_dim0):
            idx = (idx0 == i)
            new_indices = torch.cat([idx1[idx].unsqueeze(0), idx2[idx].unsqueeze(0)], dim=0)
            out.append(SpecialSpmmFunction.apply(new_indices, values[idx], shape[1:], b))
        return torch.cat(out, dim=-1) # (n_dim1, n_dim0)


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
            # nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.Linear(2*self.state_dim+8, self.hidden_unit_message_dim),
            # 2 for each hidden state, 1 for J[i,j], 1 for b[i] and 1 for b[j]
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        self.fac2var_message_passing = nn.Sequential(
            # nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.Linear(2*self.state_dim+8, self.hidden_unit_message_dim),
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
        self.spmm = Special3dSpmm()
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.fill_(0)

    # unbatch version for debugging
    def forward(self, J, b):
        n_var_nodes = len(J)
        row, col = torch.nonzero(J).t()
        n_edges = row.shape[0]
        factors = [(row[i], col[i]) for i in range(len(row)) if row[i] < col[i]]
        n_factors = len(factors)
        assert n_factors == n_edges//2

        var_hidden_states = torch.zeros(n_var_nodes, self.state_dim).to(J.device)
        fac_hidden_states = torch.zeros(n_factors,   self.state_dim).to(J.device)

        var2fac_edge_index = []
        var2fac_edge_feat = []
        fac2var_edge_index = []
        fac2var_edge_feat = []

        for i in range(len(factors)):
            # consider factor idx: n_var_nodes+i is connected to 
            # node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            var2fac_edge_index.append([u, i])
            var2fac_edge_index.append([v, i])
            fac2var_edge_index.append([i, u])
            fac2var_edge_index.append([i, v])

            var2fac_edge_feat.append([0., b[u], b[v], J[u,v]])
            var2fac_edge_feat.append([0., b[u], b[v], J[u,v]])
            fac2var_edge_feat.append([1., b[u], b[v], J[u,v]])
            fac2var_edge_feat.append([1., b[u], b[v], J[u,v]])

        var2fac_edge_index = torch.LongTensor(var2fac_edge_index).t().to(J.device)
        var2fac_edge_feat = torch.FloatTensor(var2fac_edge_feat).to(J.device)
        fac2var_edge_index = torch.LongTensor(fac2var_edge_index).t().to(J.device)
        fac2var_edge_feat  = torch.FloatTensor(fac2var_edge_feat).to(J.device)

        for step in range(self.n_steps):
            # fac2var message passing
            row, col = fac2var_edge_index
            edge_messages = torch.cat([fac_hidden_states[row, :],
                                       var_hidden_states[col, :],
                                       fac2var_edge_feat[row, :],
                                       fac2var_edge_feat[col, :]], dim=-1)


            edge_messages = self.fac2var_message_passing(edge_messages)
            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            var_hidden_states = self.fac2var_propagator(node_messages, var_hidden_states)

            # var2fac message passing
            row, col = var2fac_edge_index
            edge_messages = torch.cat([var_hidden_states[row, :],
                                       fac_hidden_states[col, :],
                                       var2fac_edge_feat[row, :],
                                       var2fac_edge_feat[col, :]], dim=-1)


            edge_messages = self.var2fac_message_passing(edge_messages)
            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            fac_hidden_states = self.var2fac_propagator(node_messages, fac_hidden_states)

        readout = self.readout(var_hidden_states)
        readout = self.softmax(readout)
        return readout
