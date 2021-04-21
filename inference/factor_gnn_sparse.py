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
            nn.Linear(self.state_dim+1, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        self.fac2var_message_passing = nn.Sequential(
            nn.Linear(self.state_dim+1, self.hidden_unit_message_dim),
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
        var2fac_hidden_states = torch.zeros(num_var2fac_msg_nodes, self.state_dim)

        fac2var_edge_index = []
        for i in range(len(factors)):
            # factor i is connected to node factors[i][0], and node factors[i][1]
            u, v = factors[i]
            fac2var_edge_index.append([i, u])
            fac2var_edge_index.append([i, v])
        num_fac2var_msg_nodes = len(fac2var_edge_index)
        fac2var_hidden_states = torch.zeros(num_fac2var_msg_nodes, self.state_dim)


        f2v_to_v2f_edge_index = []
        f2v_to_v2f_feat = torch.zeros(num_fac2var_msg_nodes, num_var2fac_msg_nodes, 1)
        v2f_to_f2v_edge_index = []
        v2f_to_f2v_feat = torch.zeros(num_var2fac_msg_nodes, num_fac2var_msg_nodes, 1)

        for ii in range(num_var2fac_msg_nodes):
            # ii is the index of var2fac_msg_node 
            # considering msg_{u -> fv}
            u, fv = var2fac_edge_index[ii]
            for jj in range(num_fac2var_msg_nodes):
                # jj is the index of fac2var msg node
                # considering msg_{fu -> v}
                fu, v = fac2var_edge_index[jj]
                if u == v and fv != fu:
                    f2v_to_v2f_edge_index.append([jj, ii])
                    f2v_to_v2f_feat[jj, ii, :] = torch.Tensor([b[u].item()])

                if fv == fu and u != v:
                    v2f_to_f2v_edge_index.append([ii, jj])
                    v2f_to_f2v_feat[ii, jj, :] = torch.Tensor([J[factors[fu][0], factors[fu][1]].item()])

        f2v_to_v2f_edge_index = torch.LongTensor(f2v_to_v2f_edge_index).t()
        v2f_to_f2v_edge_index = torch.LongTensor(v2f_to_f2v_edge_index).t()
        # var2fac_edge_feat = torch.FloatTensor(var2fac_edge_feat).to(J.device)
        # fac2var_edge_feat  = torch.FloatTensor(fac2var_edge_feat).to(J.device)

        for step in range(self.n_steps):
            # f2v_to_v2f
            # calculate var2fac messages from fac2var message nodes
            # row is the indices of fac2var message nodes
            # col is the indices of var2fac message nodes
            row, col  = f2v_to_v2f_edge_index
            raw_edge_messages = torch.cat([fac2var_hidden_states[row, :],
                                           f2v_to_v2f_feat[row, col, :]], dim=-1)
            edge_messages = self.var2fac_message_passing(raw_edge_messages)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            var2fac_hidden_states = self.var2fac_propagator(node_messages, var2fac_hidden_states)
            # we get the updated hidden_states of var2fac msg nodes

            # calculate fac2var messages from var2fac messages
            row, col = v2f_to_f2v_edge_index
            raw_edge_messages = torch.cat([var2fac_hidden_states[row, :],
                                           v2f_to_f2v_feat[row, col, :]], dim=-1)
            edge_messages = self.fac2var_message_passing(raw_edge_messages)

            node_messages = scatter(edge_messages, col, dim=0, reduce='sum')
            fac2var_hidden_states = self.fac2var_propagator(node_messages, fac2var_hidden_states)

        col = []
        for jj in range(num_fac2var_msg_nodes):
            # msg_{fu -> v}
            fu, v = fac2var_edge_index[jj]
            col.append(v)
        col = torch.LongTensor(col)

        # then use scatter to get node beliefes
        node_messages = scatter(fac2var_hidden_states, col, dim=0, reduce='sum')
        readout = self.readout(node_messages)
        readout = self.softmax(readout)
        return readout
