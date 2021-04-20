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
        # msg_dim, n_nodes, n_nodes
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


        # nn1 = Sequential(Linear(2*self.state_dim+1+2, self.hidden_unit_message_dim),
                         # ReLU(),
                         # Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim))
        # nn2 = Sequential(Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
                         # ReLU(),
                         # Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim))
        # self.convs = [GINConv(nn1), GINConv(nn2)]

        self.conv = GatedGraphConv(self.hidden_unit_message_dim, 2)

        # self.propagator = nn.GRUCell(self.message_dim, self.state_dim)
        self.message_passing = nn.Sequential(
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            # nn.Linear(2*self.state_dim+1+2, self.hidden_unit_message_dim),
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
        row, col = torch.nonzero(J).t()
        n_nodes = len(J)
        n_edges = row.shape[0]
        readout = torch.zeros(n_nodes)
        # initialize node embeddings to zeros
        variable_nodes_feat = torch.zeros(n_nodes, self.state_dim).to(J.device)
        variable_nodes_feat[:, 0] = 0.
        
        factor_nodes_feat = torch.zeros(n_nodes + n_edges, self.state_dim)
        factor_nodes_feat[:, 0] = 1.
        factor_nodes_feat[:, 1] = J[row, col]

        edge_index = []
        for i in range(n_node):
            edge_index.append([i, n_nodes+i])
            edge_index.append([n_nodes+i, i])
        for i in range(n_edges):
            # consider factor node idx: n_nodes+i
            edge_index.append([n_nodes+i, row[i]])
            edge_index.append([row[i], n_nodes+i])
            edge_index.append([n_nodes+i, col[i]])
            edge_index.append([col[i], n_nodes+i])
        edge_index = torch.LongTensor(edge_index).t().to(J.device)
        # import ipdb;ipdb.set_trace()
        # for i in range(len(self.convs)):
            # edge_messages = self.convs[i](edge_messages, edge_index)
        edge_messages = self.conv(edge_messages, edge_index)

        edge_messages = self.message_passing(edge_messages).t().reshape(-1) # in message, (dim2*dim0*dim1)
        edges = torch.nonzero(J.unsqueeze(-1).expand(-1, -1, self.message_dim).permute(2,0,1)).t()# (dim2*dim0*dim1) 
        node_messages = self.spmm(edges,
                                  edge_messages,
                                  torch.Size([self.message_dim, n_nodes, n_nodes]),
                                  torch.ones(size=(n_nodes,1)).to(J.device)) # (dim0, dim2)

        readout = self.readout(node_messages)
        # readout = self.readout(hidden_states)
        readout = self.softmax(readout)
        # readout = self.sigmoid(readout)
        # readout = readout / torch.sum(readout,1).view(-1,1)
        return readout
