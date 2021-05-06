#!/usr/bin/env python
# coding=utf-8

import numpy as np

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import scatter_
from torch_geometric.data import Batch
from torch_scatter import scatter_logsumexp, scatter_max, scatter_sum, scatter_min, scatter_mean

from inference.core import Inference
from inference.gnn_inference import GatedGNNInference

LN_ZERO = -1e11

###################################################
############ Build the Interface ##################
###################################################

def build_padded_factor_potential(padded_size, log_factor_potential, mask):
    '''
    Inputs:
    - padded_size (list of ints): the output size after padding
    - factor_potential (tensor): the input factor potential before padding
    - mask (tensor): the input factor potential mask before padding

    Outputs:
    - padded_factor_potential (tensor): 1 for variable assignments that satisfy the clause, 0 otherwise
    - padded_mask (tensor): 1 signifies an invalid location (outside factor's valid variables), 0 signifies a valid location
    '''
    input_dimensions = len(log_factor_potential.shape) #dimensions in the input factor potential
    padded_dimensions = len(padded_size) #dimensions in the output factor potential after padding
    assert(padded_dimensions > input_dimensions)

    assert(tuple(log_factor_potential.shape) == tuple(padded_size[:input_dimensions]))
    padded_log_factor_potential = torch.zeros(padded_size)+LN_ZERO
    padded_mask = torch.zeros(padded_size)

    for indices in np.ndindex(padded_log_factor_potential.shape):
        junk_location = False
        for dimension in range(input_dimensions, padded_dimensions):
            if indices[dimension] >= 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            padded_mask[indices] = 1
            continue
        else:
            padded_mask[indices] = mask[indices[:input_dimensions]]
            padded_log_factor_potential[indices] = log_factor_potential[indices[:input_dimensions]]
    return padded_log_factor_potential, padded_mask

'''
Convert Binary MRF (represented by J and b) into FactorGraphData
'''
def build_FactorGraph_from_BMRF(J, b):
    '''
    J of shape [n_V, n_V] and b of shape [n_V]
    '''
    variable_count = b.shape[0]
    variable_cardinalities = [2,]*variable_count
    var_cardinality = max(variable_cardinalities)

    factorToVar_edge_index = []
    edge_var_indices = []
    factorToVar_double_list = []
    log_factor_potentials = []
    factor_potential_masks = []
    #the number of variables in the largest factor
    max_factor_dimension = 0
    all_factor_dimensions = [] #all_factor_dimensions[i] is the number of variables in the ith factor

    trow, tcol = torch.nonzero(J).t()
    row, col = [], []
    for r,c in zip(trow, tcol):
        if r.item() <= c.item():
            row.append(r)
            col.append(c)
    row = torch.LongTensor(row)
    col = torch.LongTensor(col)
    assert row.shape[0] == col.shape[0] and row.shape[0] == trow.shape[0]//2

    binary_factor_count, unary_factor_count = row.shape[0], variable_count
    factor_count = binary_factor_count+unary_factor_count

    for factor_idx in range(unary_factor_count):
        factor_dimension = 1
        all_factor_dimensions.append(factor_dimension)
        if factor_dimension > max_factor_dimension:
            max_factor_dimension = factor_dimension
        var_indices = [factor_idx,]
        assert(factor_dimension == len(var_indices))
        for var_idx_in_factor, var_idx in enumerate(var_indices):
            factorToVar_edge_index.append([factor_idx, var_idx])
            edge_var_indices.append([var_idx_in_factor, -99])
        factorToVar_double_list.append(var_indices)

        factor_entry_count = 2
        tmp_local_potential = b[factor_idx]
        log_factor_entries = [-tmp_local_potential, tmp_local_potential,]
        assert(factor_entry_count == len(log_factor_entries))
        factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in var_indices]
        log_factor_potential = torch.tensor(log_factor_entries).reshape(factor_var_cardinalities)
        factor_potential_mask = (torch.isinf(log_factor_potential)+(log_factor_potential<=LN_ZERO) > 0).float()
        for fvidx, fvcard in enumerate(factor_var_cardinalities):
            factor_var_cardinalities[fvidx] = var_cardinality - factor_var_cardinalities[fvidx]
            log_factor_potential = torch.cat([log_factor_potential, LN_ZERO+torch.zeros(factor_var_cardinalities)], dim=fvidx)
            factor_potential_mask = torch.cat([factor_potential_mask, torch.ones(factor_var_cardinalities)], dim=fvidx)
            factor_var_cardinalities[fvidx] = var_cardinality
        log_factor_potentials.append(log_factor_potential)
        factor_potential_masks.append(factor_potential_mask)

    for factor_idx in range(binary_factor_count):
        factor_dimension = 2
        all_factor_dimensions.append(factor_dimension)
        if factor_dimension > max_factor_dimension:
            max_factor_dimension = factor_dimension
        var_indices = [row[factor_idx], col[factor_idx]]
        assert(factor_dimension == len(var_indices))
        for var_idx_in_factor, var_idx in enumerate(var_indices):
            factorToVar_edge_index.append([factor_idx+unary_factor_count, var_idx])
            edge_var_indices.append([var_idx_in_factor, -99])
        factorToVar_double_list.append(var_indices)

        factor_entry_count = 4
        u, v = var_indices[0].item(), var_indices[1].item()
        A, B = J[u,v], J[v,u]
        # tmp_local_potential = J[var_indices]
        # log_factor_entries = [tmp_local_potential, -tmp_local_potential,
                              # -tmp_local_potential, tmp_local_potential]
        log_factor_entries = [A+B, -2*A,
                              -2*B, A+B]
        assert(factor_entry_count == len(log_factor_entries))
        factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in var_indices]
        log_factor_potential = torch.tensor(log_factor_entries).reshape(factor_var_cardinalities)
        factor_potential_mask = (torch.isinf(log_factor_potential)+(log_factor_potential<=LN_ZERO) > 0).float()
        for fvidx, fvcard in enumerate(factor_var_cardinalities):
            factor_var_cardinalities[fvidx] = var_cardinality - factor_var_cardinalities[fvidx]
            log_factor_potential = torch.cat([log_factor_potential, LN_ZERO+torch.zeros(factor_var_cardinalities)], dim=fvidx)
            factor_potential_mask = torch.cat([factor_potential_mask, torch.ones(factor_var_cardinalities)], dim=fvidx)
            factor_var_cardinalities[fvidx] = var_cardinality
        log_factor_potentials.append(log_factor_potential)
        factor_potential_masks.append(factor_potential_mask)

    padded_log_factor_potentials = [] #pad factor potentials so they all have the same size
    no_padded_log_factor_potentials = [] #pad factor potentials so they all have the same size
    # factor_potential_masks = [] #store masks: 1 signifies an invalid location (padding), 0 signifies a valid location
    for factor_idx, log_factor_potential in enumerate(log_factor_potentials):
        if all_factor_dimensions[factor_idx] == max_factor_dimension:
            padded_log_factor_potentials.append(log_factor_potential)
            no_padded_log_factor_potentials.append(log_factor_potential)
            # factor_potential_masks.append(torch.zeros([var_cardinality for i in range(max_factor_dimension)]))
        else:
            padded_size = [var_cardinality for i in range(max_factor_dimension)]
            padded_log_factor_potential, padded_mask = build_padded_factor_potential(
                padded_size=padded_size, log_factor_potential=log_factor_potential,
                mask=factor_potential_masks[factor_idx],
            )
            no_padded_log_factor_potential = torch.zeros_like(padded_log_factor_potential)
            index = (slice(None),)*all_factor_dimensions[factor_idx] + (0,)*(max_factor_dimension-all_factor_dimensions[factor_idx])
            no_padded_log_factor_potential[index] = padded_log_factor_potential[index]

            padded_log_factor_potentials.append(padded_log_factor_potential)
            factor_potential_masks[factor_idx] = padded_mask
            no_padded_log_factor_potentials.append(no_padded_log_factor_potential)

    log_factor_potentials = torch.stack(padded_log_factor_potentials, dim=0)
    no_padded_log_factor_potentials = torch.stack(no_padded_log_factor_potentials, dim=0)
    factor_potential_masks = torch.stack(factor_potential_masks, dim=0)
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index).t().contiguous()
    edge_var_indices = torch.tensor(edge_var_indices).t().contiguous()
    mpe = LN_ZERO

    # log_factor_potentials = torch.log(factor_potentials) #@hao: filter out infinitiy at the first place
    log_factor_potentials[torch.isinf(log_factor_potentials)] = LN_ZERO
    no_padded_log_factor_potentials[torch.isinf(no_padded_log_factor_potentials)] = LN_ZERO
    factor_graph = FactorGraphData(
        factor_potentials=log_factor_potentials,
        no_padded_factor_potentials=no_padded_log_factor_potentials,
        factorToVar_edge_index=factorToVar_edge_index, numVars=variable_count, numFactors=factor_count,
        edge_var_indices=edge_var_indices, state_dimensions=max_factor_dimension,
        factor_potential_masks=factor_potential_masks, mpe=mpe,
        factorToVar_double_list=factorToVar_double_list,
        var_cardinality=var_cardinality,
    )
    return factor_graph

class MaxBPNN(nn.Module):
    def __init__(self, mode='map', damping=0., normalization_flag=True,
                 hidden_size=64, batch_norm_flag=True,
                 maxiter=10000, tol=1e-9,
                 src_names=['diff_f2v_messages', 'diff_beliefs'],):
        super(MaxBPNN, self).__init__()
        if mode not in ["marginal", "map"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.maxbpnn_model = ScheduleMaxBPV2(
            mode = mode,
            damping=damping, normalization_flag=normalization_flag,
            hidden_size=hidden_size, batch_norm_flag=batch_norm_flag,
            maxiter=maxiter, tol=tol,
            src_names=src_names,
        )
    def forward(self, J, b):
        graph = build_FactorGraph_from_BMRF(J, b)
        batched_graphs = Batch.from_data_list([graph,])
        var_beliefs, _ = self.maxbpnn_model(batched_graphs)
        return var_beliefs

class MaxBPNNInference(GatedGNNInference):
    def __init__(self, mode, state_dim, message_dim,
                hidden_unit_message_dim, hidden_unit_readout_dim,
                n_steps=10, load_path=None, sparse=True):
        Inference.__init__(self, mode)

        normalization_flag=False
        damping=.999
        print('normalization_flag', normalization_flag)
        print('damping', damping)

        self.model = MaxBPNN(
            mode=mode,
            hidden_size=hidden_unit_message_dim,
            damping=damping, normalization_flag=normalization_flag, #@hao: These four hyper-parameters are also important for the performance. Espeically the damping and the normalization flag.
            batch_norm_flag=False, maxiter=200,     #@hao: Probably we should modify the interface in train.py to tune these hyper-parameters?
        )
        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))
            self.model.eval()
        self.history = {"loss": []}
        self.batch_size = 50

###################################################
############ Define FactorGraph ###################
###################################################

from torch_geometric.data import Data

def create_scatter_indices_helper(expansion_index, variable_cardinality, state_dimensions, offset):
    '''
    Inputs:
    - expansion_index (int):
    -
    -
    - offset (int): add this value to every index
    '''
    l = torch.tensor([i + offset for i in range(variable_cardinality**state_dimensions)])
    l_shape = [variable_cardinality for i in range(state_dimensions)]
    l = l.reshape(l_shape)
    l = l.transpose(expansion_index, 0)
    return l.flatten()

'''
@hao: It seems that this indices are useful when sending variable messages to factors.
The variable messages have shape [numMsgs, var_cardinality], and
the factor messages have shape [numMsgs, var_cardinality**state_dimensions].

The variable messages are fist expanded to shape [numMsgs, var_cardinality**state_dimensions],
and then tranposed/permuted by this indices for correct dimensions.
'''
def create_scatter_indices_varToFactorMsgs(original_indices, variable_cardinality=2, state_dimensions=2):
    #When sending variable to factor messages, variable beliefs must be expanded to have extra redundant dimensions.
    #The approach is to expand all variable beliefs in all outgoing messages, then transpose each belief appropriately
    #so that the variable it represents lines up in the correct dimension of the factor each message is sent to.
    #This function creates indices, to be used with torch_scatter, to perform this functionality.
    assert((original_indices < state_dimensions).all())
    scatter_indices_list = []
    for position, index in enumerate(original_indices):
        cur_offset = position*(variable_cardinality**state_dimensions)
        cur_indices = create_scatter_indices_helper(
            expansion_index=index, variable_cardinality=variable_cardinality,
            state_dimensions=state_dimensions, offset=cur_offset
        )
        scatter_indices_list.append(cur_indices)
    scatter_indices = torch.cat(scatter_indices_list)
    return scatter_indices

class FactorGraphData(Data):
    '''
    Representation of a factor graph in pytorch geometric Data format
    '''
    def __init__(
        self, factor_potentials=None, factorToVar_edge_index=None, numVars=None, numFactors=None,
        edge_var_indices=None, state_dimensions=None, factor_potential_masks=None, mpe=None,
        factorToVar_double_list=None, var_cardinality=None,
        no_padded_factor_potentials=None,
    ):
        '''
        Inputs:
        - factor_potentials (torch tensor): represents all factor potentials (log base e space) and has
            shape (num_factors, var_states, ..., var_states), where var_states is the number
            of states a variable can take (e.g. 2 for an ising model or 3 for community detection
            with 3 communities).  Has state_dimensions + 1 total dimensions (where state_dimensions
            is the number of dimenions in the largest factor), e.g. 3 total
            dimensions when the largest factor contains 2 variables.  Factors with fewer than
            state_dimensions dimensions have unused states set to -infinitiy (0 probability)
        - factorToVar_edge_index (torch tensor): shape (2, num_edges), represents all edges in the factor graph.
            factorToVar_edge_index[0, i] is the index of the factor connected by the ith edge
            factorToVar_edge_index[1, i] is the index of the variable connected by the ith edge
        - numVars (int): number of variables in the factor graph
        - numFactors (int): number of factors in the factor graph
        - edge_var_indices (torch tensor): shape (2, num_edges)
            [0, i] indicates the index (0 to factor_degree - 1) of edge i, among all edges originating at the factor which edge i begins at
            [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicate the index (0 to var_degree - 1) of edge i, among all edges ending at the variable which edge i ends at
        - state_dimensions (int): the number of variables in the largest factor
        - factor_potential_masks (torch tensor): same shape as factor_potentials.  All entries are 0 or 1.
            0: represents that the corresponding location in factor_potentials is valid
            1: represents that the corresponding location in factor_potentials is invalid and should be masked,
               e.g. the factor has fewer than state_dimensions dimensions
        - mpe: natural logarithm of the value of MPE/MAP solution
        - factorToVar_double_list (list of lists):
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with

        - var_cardinality (int): variable cardinality, the number of states each variable can take.
        '''
        self.LN_ZERO = -1e33
        super(FactorGraphData, self).__init__()
        if mpe is not None:
            self.mpe = torch.tensor([mpe], dtype=float)

        # (int) the largest node degree
        if state_dimensions is not None:
            self.state_dimensions = torch.tensor([state_dimensions])
        else:
            self.state_dimensions = None
        self.var_cardinality = var_cardinality

        self.factor_potentials = factor_potentials
        self.no_padded_factor_potentials = no_padded_factor_potentials
        self.factor_potential_masks = factor_potential_masks
        self.factorToVar_double_list = factorToVar_double_list

        # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        #     matrix with shape :obj:`[numFactors, numVars]`
        #     stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge
        if factorToVar_double_list is not None:
            #facStates_to_varIdx (torch LongTensor): essentially representing edges and pseudo edges (between
            #    junk states and a junk bin)
            #    has shape [(number of factor to variable edges)*(2^state_dimensions]  with values
            #    in {0, 1, ..., (number of factor to variable edges)-1, (number of factor to variable edges)}.
            #    Note (number of factor to variable edges) indicates a 'junk state' and should be output into a
            #    'junk' bin after scatter operation.
            self.facStates_to_varIdx, self.facToVar_edge_idx = self.create_factorStates_to_varIndices(factorToVar_double_list)
#             self.facStates_to_varIdx_FIXED, self.facToVar_edge_idx_FIXED = self.create_factorStates_to_varIndices_FIXED(factorToVar_double_list)
        elif factorToVar_edge_index is not None:
            if factorToVar_double_list is not None:
                assert((self.facToVar_edge_idx == factorToVar_edge_index).all())
            else:
                self.facToVar_edge_idx = factorToVar_edge_index
                self.facStates_to_varIdx = None
        else:
            self.facToVar_edge_idx = None
            self.facStates_to_varIdx = None
        self.edge_index = self.facToVar_edge_idx #hack for batching, see learn_BP_spinGlass.py

        if factorToVar_edge_index is not None:
            # self.factor_degrees[i] stores the number of variables that appear in factor i
            unique_factor_indices, self.factor_degrees = torch.unique(factorToVar_edge_index[0,:], sorted=True, return_counts=True)
            assert((self.factor_degrees >= 1).all())
            assert(unique_factor_indices.shape[0] == numFactors), (unique_factor_indices.shape[0], numFactors)
            # self.var_degrees[i] stores the number of factors that variables i appears in
            unique_var_indices, self.var_degrees = torch.unique(factorToVar_edge_index[1,:], sorted=True, return_counts=True)
            assert((self.var_degrees >= 1).all())
            assert(unique_var_indices.shape[0] == numVars)
        else:
            self.factor_degrees = None
            self.var_degrees = None

        #when batching, numVars and numFactors record the number of variables and factors for each graph in the batch
        if numVars is not None:
            self.numVars = torch.tensor([numVars])
        else:
            self.numVars = None
        if numFactors is not None:
            self.numFactors = torch.tensor([numFactors])
        else:
            self.numFactors = None
        #when batching, see num_vars and num_factors to access the cumulative number of variables and factors in all
        #graphs in the batch, like num_nodes

        # edge_var_indices has shape [2, E].
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i,
        #       among all edges originating at the node which edge i begins at
        #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
        self.edge_var_indices = edge_var_indices
        if edge_var_indices is not None:
            assert(self.facToVar_edge_idx.shape == self.edge_var_indices.shape)
            self.varToFactorMsg_scatter_indices = create_scatter_indices_varToFactorMsgs(
                original_indices=self.edge_var_indices[0, :],
                variable_cardinality=self.var_cardinality,
                state_dimensions=state_dimensions,
            )
        else:
            self.varToFactorMsg_scatter_indices = None

        if self.var_cardinality is not None:
            prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = self.get_initial_beliefs_and_messages()
            self.prv_varToFactor_messages = prv_varToFactor_messages
            self.prv_factorToVar_messages = prv_factorToVar_messages
            self.prv_factor_beliefs = prv_factor_beliefs
            self.prv_var_beliefs = prv_var_beliefs
            assert(self.prv_factor_beliefs.size(0) == self.numFactors)
            assert(self.prv_var_beliefs.size(0) == self.numVars)
        else:
            self.prv_varToFactor_messages = None
            self.prv_factorToVar_messages = None
            self.prv_factor_beliefs = None
            self.prv_var_beliefs = None

        # cache the masks for beliefs and messages.
        # mask = 0 indicates that the corresponding position is valid,
        # otherwise mask = 1
        self.factor_beliefs_masks = self.factor_potential_masks.bool()
        mapped_factor_beliefs_masks = self.factor_beliefs_masks[self.facToVar_edge_idx[0]] #map node beliefs to edges
        factorToVar_messages_masks = scatter_min(
            src=mapped_factor_beliefs_masks.reshape([-1]).float()[self.facStates_to_varIdx>=0],
            index=self.facStates_to_varIdx[self.facStates_to_varIdx>=0],
            dim_size=self.edge_index.size(1)*self.var_cardinality,
        )[0].bool()
        self.factorToVar_messages_masks = factorToVar_messages_masks.reshape(
            mapped_factor_beliefs_masks.shape[:2]
        )
        self.var_beliefs_masks = scatter_(
            'add', self.factorToVar_messages_masks.float(),
            self.facToVar_edge_idx[1], dim_size=self.num_vars
        ).bool()
        self.varToFactor_messages_masks = self.var_beliefs_masks[self.facToVar_edge_idx[1]] #map node beliefs to edges


    def __inc__(self, key, value):
        if key == 'facStates_to_varIdx':
            return torch.tensor([self.var_cardinality*self.edge_var_indices.size(1)])
        elif key == 'varToFactorMsg_scatter_indices':
            return torch.tensor([self.varToFactorMsg_scatter_indices.size(0)])
        elif key == 'edge_index' or key == 'facToVar_edge_idx':
            return torch.tensor([self.prv_factor_beliefs.size(0), self.prv_var_beliefs.size(0)]).unsqueeze(dim=1)
        elif key == 'edge_var_indices':
            return torch.tensor([0, 0]).unsqueeze(dim=1)
        else:
            return super(FactorGraphData, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if key == 'facToVar_edge_idx' or key == 'edge_var_indices':
            return -1
        else:
            return super(FactorGraphData, self).__cat_dim__(key, value)


    def create_factorStates_to_varIndices(self, factorToVar_double_list):
        '''
        Inputs:
        - factorToVar_double_list (list of lists):
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with

        Output:
        - factorStates_to_varIndices (torch LongTensor):
            shape [(number of factor to variable edges)*(self.var_cardinality^state_dimensions]
            with values in {0, 1, ..., self.var_cardinality*(number of factor to variable edges)-1, -1}.
            Note -1 indicates a 'junk state' and should be output into a
            'junk' bin after scatter operation.
            Used with scatter_logsumexp (https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
            to marginalize the appropriate factor states for each message

        - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge matrix
            stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge
        '''
        # the number of (undirected) edges in the factor graph, or messages in one graph wide update
        numMsgs = sum([len(variables_list) for variables_list in factorToVar_double_list])

        factorStates_to_varIndices_list = []
        factorToVar_edge_index_list = []

        # junk_bin = -1 #new junk bin for batching
        junk_bin = -1e9 #new junk bin for my own batching

        arange_tensor = torch.arange(self.var_cardinality**self.state_dimensions.item())
        msg_idx = 0
        for factor_idx, variables_list in enumerate(factorToVar_double_list):
            unused_var_count = self.state_dimensions - len(variables_list)
            for varIdx_inFac, var_idx in enumerate(variables_list):
                factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))

                curFact_to_varIndices = -99*torch.ones(self.var_cardinality**self.state_dimensions, dtype=torch.long)

                multiplier1 = self.var_cardinality**(self.state_dimensions - varIdx_inFac - 1)
                for var_state_idx in range(self.var_cardinality):
                    curFact_to_varIndices[((arange_tensor//multiplier1) % self.var_cardinality) == var_state_idx] = msg_idx + var_state_idx
                assert(not (curFact_to_varIndices == -99).any())

                #send unused factor states to the junk bin
                if unused_var_count > 0:
                    multiplier2 = 1 #note multiplier2 doubles at each iteration, looping over the variables backwards compared to multiplier1
                    for unused_var_idx in range(unused_var_count):
                        #send all factor states correpsonding to the unused variable being in any state except 0 to the junk bin
                        for var_state_idx in range(1, self.var_cardinality):
                            curFact_to_varIndices[((arange_tensor//multiplier2) % self.var_cardinality) == var_state_idx] = junk_bin
                        multiplier2 *= self.var_cardinality

                # for repeat_idx in range(self.belief_repeats):
                    # curRepeat_curFact_to_varIndices = curFact_to_varIndices.clone()
                factorStates_to_varIndices_list.append(curFact_to_varIndices)
                msg_idx += self.var_cardinality
                    # curFact_to_varIndices[torch.where(curFact_to_varIndices != -1)] += self.var_cardinality

        assert(msg_idx == self.var_cardinality*numMsgs), (msg_idx, numMsgs, self.var_cardinality*numMsgs)
        assert(len(factorStates_to_varIndices_list) == numMsgs)
        factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list, dim=0)
        factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
        return factorStates_to_varIndices, factorToVar_edge_index

    def get_initial_beliefs_and_messages(self, initialize_randomly=False, device=None):
        edge_count = self.edge_var_indices.shape[1]

        prv_varToFactor_messages = torch.log(torch.stack([
            torch.ones([self.var_cardinality], dtype=torch.float)
            for j in range(edge_count)], dim=0))
        prv_factorToVar_messages = torch.log(torch.stack([
            torch.ones([self.var_cardinality], dtype=torch.float)
            for j in range(edge_count)], dim=0))
        prv_factor_beliefs = torch.log(torch.stack([
            torch.ones([self.var_cardinality for i in range(self.state_dimensions)], dtype=torch.float)
            for j in range(self.numFactors)], dim=0))
        prv_var_beliefs = torch.log(torch.stack([
            torch.ones([self.var_cardinality], dtype=torch.float)
            for j in range(self.numVars)], dim=0))
        if initialize_randomly:
            prv_varToFactor_messages = torch.rand_like(prv_varToFactor_messages)
            prv_factorToVar_messages = torch.rand_like(prv_factorToVar_messages)
            # prv_factor_beliefs = torch.rand_like(prv_factor_beliefs)
            prv_var_beliefs = torch.rand_like(prv_var_beliefs)
        if device is not None:
            prv_varToFactor_messages = prv_varToFactor_messages.to(device)
            prv_factorToVar_messages = prv_factorToVar_messages.to(device)
            prv_factor_beliefs = prv_factor_beliefs.to(device)
            prv_var_beliefs = prv_var_beliefs.to(device)

        #These locations are unused, set to LN_ZERO as a safety check for assert statements elsewhere
        assert(prv_factor_beliefs.shape == self.factor_potential_masks.shape), (prv_factor_beliefs.shape, self.factor_potential_masks.shape)
        prv_factor_beliefs[torch.where(self.factor_potential_masks==1)] = self.LN_ZERO

        return prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs

    def _logScore(self, state):
        '''
        input args:
            state: type tuple/list, shape [numVars], element type int.
        return:
            logscore: pytorch-tensor of shape [], type float

        '''
        # logscore = 0.
        # for fidx, var_list in enumerate(self.factorToVar_double_list):
            # logscore = logscore + self.factor_potentials[index]
        index_list = [
            [fidx,] + [state[vidx] for vidx in var_list] + [0]*(self.state_dimensions-len(var_list))
            for fidx, var_list in enumerate(self.factorToVar_double_list)
        ]
        index_list = list(zip(*index_list)) #transpose
        flag = torch.sum(self.factor_potential_masks[index_list].float()) > 0
        if flag:
            return torch.tensor([-np.inf,], dtype=torch.float, device=flag.device).reshape([])
        else:
            logscore = torch.sum(self.factor_potentials[index_list]).reshape([])
            return logscore

    def logScore_middle(self, state):
        '''
        input args:
            # state: pytorch-tensor of shape [numVars, var_cardinality], type float
            state: type tuple/list, shape [numVars], element type int.
        return:
            logscore: pytorch-tensor of shape [], type float
        '''
        state = torch.eye(self.var_cardinality)[list(state)]
        edge_state = state[self.facToVar_edge_idx[1]] #[numMsgs, var_cardinality]
        # expanded_edge_state = torch.zeros(
            # [edge_state.size(0),] + [edge_state.size(1),]*self.state_dimensions.item(),
            # dtype=edge_state.dtype, device=edge_state.device,
        # )
        # expanded_edge_state[(slice(None),)*len(edge_state.shape)+(0,)*(self.state_dimensions.item()-len(edge_state.shape)+1)] = edge_state
        expanded_edge_state = edge_state[(...,)+(None,)*(self.state_dimensions.item()-1)] #[numMsgs, var_cardinality]+[1,]*(state_dimensions-1)
        expanded_edge_state = expanded_edge_state.expand(
            [edge_state.size(0),] + [edge_state.size(1),]*self.state_dimensions.item()
        )#[numMsgs,]+ [var_cardinality]*state_dimensions
        permuted_expanded_edge_state = expanded_edge_state.flatten()[
            self.varToFactorMsg_scatter_indices].reshape(expanded_edge_state.shape)#[numMsgs,]+ [var_cardinality]*state_dimensions
        factor_state = scatter_(
            'min', permuted_expanded_edge_state, self.facToVar_edge_idx[0],
            dim_size=self.num_factors
        ) #[numfactors,]+ [var_cardinality]*state_dimensions

        # compensate for padded potentials
        factor_state_shape = factor_state.shape
        flattened_factor_state = factor_state.reshape([factor_state_shape[0],-1])
        flattened_factor_state = flattened_factor_state * torch.arange(flattened_factor_state.shape[-1], 0, -1).unsqueeze(0) #find first nonzero index (original argmax will return the last nonzero index)
        factor_state = torch.eye(flattened_factor_state.shape[-1])[torch.argmax(flattened_factor_state, dim=-1)]
        factor_state = factor_state.reshape(factor_state_shape)
        logscore = torch.sum(factor_state*self.factor_potentials)
        return logscore

    def logScore(self, state):
        return self.logScore_parallel([state])[0]
    def logScore_parallel(self, state_list):
        '''
        input args:
            state: type tuple/list, shape [numVars], element type int.
        return:
            logscore: pytorch-tensor of shape [], type float
        '''
        num_states, num_vars = len(state_list), len(state_list[0])
        state_list = torch.stack([
            torch.eye(self.var_cardinality, device=self.facToVar_edge_idx.device)[list(state)]
            for state in state_list
        ], dim=-1)
        edge_state_list = state_list[self.facToVar_edge_idx[1]] #[numMsgs, var_cardinality, numStates]
        expanded_edge_state_list = edge_state_list[(slice(None),)*2+(None,)*(self.state_dimensions.item()-1)] #[numMsgs, var_cardinality]+[1,]*(state_dimensions-1)+[numStates]
        expanded_edge_state_list = expanded_edge_state_list.expand(
            [edge_state_list.size(0),] + [edge_state_list.size(1),]*self.state_dimensions.item() + [num_states,]
        )#[numMsgs,]+ [var_cardinality]*state_dimensions +[numStates]
        permuted_expanded_edge_state_list = expanded_edge_state_list.reshape([-1, num_states])[
            self.varToFactorMsg_scatter_indices].reshape(expanded_edge_state_list.shape)#[numMsgs,]+ [var_cardinality]*state_dimensions +[numStates]
        factor_state_list = scatter_(
            'min',
            permuted_expanded_edge_state_list,
            self.facToVar_edge_idx[0],
            dim_size=self.num_factors
        ) #[numfactors,]+ [var_cardinality]*state_dimensions +[numStates]

        # compensate for padded potentials
        factor_state_list = factor_state_list.permute(
            *((self.state_dimensions.item()+1,)+
              tuple(range(self.state_dimensions.item()+1)))
        ) #[numStates, numfactors,]+ [var_cardinality]*state_dimensions
        factor_state_list_shape = factor_state_list.shape
        flattened_factor_state = factor_state_list.reshape(
            [num_states*factor_state_list_shape[1],-1])
        flattened_factor_state = flattened_factor_state * torch.arange(
            flattened_factor_state.shape[-1], 0, -1,
            device=self.facToVar_edge_idx.device,
        ).unsqueeze(0)#find first nonzero index (original argmax will return the last nonzero index)
        factor_state = torch.eye(
            flattened_factor_state.shape[-1],
            device=self.facToVar_edge_idx.device,
        )[torch.argmax(flattened_factor_state, dim=-1)]
        factor_state_list = factor_state.reshape(factor_state_list_shape)
        logscore = torch.sum(factor_state_list*self.factor_potentials.unsqueeze(0),
                             dim=tuple(range(1,len(factor_state_list_shape))))
        return logscore

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
#         if self.edge_index is not None:
#             warnings.warn(__num_nodes_warn_msg__.format('edge'))
#             return maybe_num_nodes(self.edge_index)
        return None
    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes
    @property
    def num_vars(self):
        r"""Returns or sets the number of variables in the factor graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_vars = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, 'prv_factor_beliefs'):
            return self.prv_var_beliefs.size(self.__cat_dim__('prv_factor_beliefs', self.prv_var_beliefs))
        else:
            return None
    @num_vars.setter
    def num_vars(self, num_vars):
        self.__num_vars__ = num_vars
    @property
    def num_factors(self):
        r"""Returns or sets the number of factors in the factor graph.
        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_factors = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, 'prv_factor_beliefs'):
            return self.prv_factor_beliefs.size(self.__cat_dim__('prv_factor_beliefs', self.prv_factor_beliefs))
        else:
            return None
    @num_factors.setter
    def num_factors(self, num_factors):
        self.__num_factors__ = num_factors


###################################################
############### Define MaxBPNN ####################
###################################################

def max_multipleDim(input, axes, keepdim=False):
    '''
    modified from https://github.com/pytorch/pytorch/issues/2006
    Take the maximum across multiple axes.  For example, say the input has dimensions [a,b,c,d] and
    axes = [0,2].  The output (with keepdim=False) will have dimensions [b,d] and output[i,j] will
    be the maximum value of input[:, i, :, j]
    '''
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.max(ax, keepdim=True)[0]
    else:
        for ax in sorted(axes, reverse=True):
            input = input.max(ax)[0]
    return input
def logsumexp_multipleDim(tensor, dim=None):
    """
    Compute log(sum(exp(tensor), dim)) in a numerically stable way.

    Inputs:
    - tensor (tensor): input tensor
    - dim (int): the only dimension to keep in the output.  i.e. for a 4d input tensor with dim=2 (0-indexed):
        return_tensor[i] = logsumexp(tensor[:,:,i,:])

    Outputs:
    - return_tensor (1d tensor): logsumexp of input tensor along specified dimension

    """
    assert(not torch.isnan(tensor).any())

    tensor_dimensions = len(tensor.shape)
    assert(dim < tensor_dimensions and dim >= 0)
    aggregate_dimensions = [i for i in range(tensor_dimensions) if i != dim]
    max_values = max_multipleDim(tensor, axes=aggregate_dimensions, keepdim=True)
    max_values[torch.where(max_values == -np.inf)] = 0
    assert(not torch.isnan(max_values).any())
    assert((max_values > -np.inf).any())
    assert(not torch.isnan(tensor - max_values).any())
    assert(not torch.isnan(torch.exp(tensor - max_values)).any())
    assert(not torch.isnan(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)).any())
    assert(not torch.isnan(torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions))).any())
    return_tensor = torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)) + max_values.squeeze()
    assert(not torch.isnan(return_tensor).any())
    return return_tensor
def cal_var_beliefs(factor_graph, factorToVar_messages):
    var_beliefs = scatter_(
        'add', factorToVar_messages,
        factor_graph.facToVar_edge_idx[1],
        dim_size=torch.sum(factor_graph.numVars).item(),
    ) #sum of incoming fac2var messages
    var_beliefs[torch.where(factor_graph.var_beliefs_masks)] = LN_ZERO
    var_beliefs = torch.clamp(var_beliefs, min=LN_ZERO)
    return var_beliefs
def cal_varToFactor_messages(factor_graph, var_beliefs, factorToVar_messages):
    mapped_var_beliefs = var_beliefs[factor_graph.facToVar_edge_idx[1]] #map node beliefs to edges
    varToFactor_messages = mapped_var_beliefs-factorToVar_messages #avoid double counting
    varToFactor_messages[torch.where(factor_graph.varToFactor_messages_masks)] = LN_ZERO
    varToFactor_messages = torch.clamp(varToFactor_messages, min=LN_ZERO)
    return varToFactor_messages
def expand_varToFactor_messages(factor_graph, varToFactor_messages):
    state_dimensions = factor_graph.state_dimensions[0].item()
    expanded_varToFactor_messages_shape = list(varToFactor_messages.shape)+[
        factor_graph.var_cardinality[0].item()
    ]*(state_dimensions-1)
    expanded_varToFactor_messages = varToFactor_messages[(...,)+(None,)*(state_dimensions-1)] #expand var2fac messages
    expanded_varToFactor_messages = expanded_varToFactor_messages.expand(expanded_varToFactor_messages_shape)
    expanded_varToFactor_messages = expanded_varToFactor_messages.flatten()[factor_graph.varToFactorMsg_scatter_indices]
    expanded_varToFactor_messages = expanded_varToFactor_messages.reshape(expanded_varToFactor_messages_shape) #transpose expanded messages to aligh the indexes of variables in factors
    return expanded_varToFactor_messages
def cal_factor_beliefs(factor_graph, expanded_varToFactor_messages):
    factor_beliefs = scatter_(
        'add', expanded_varToFactor_messages,
        factor_graph.facToVar_edge_idx[0],
        dim_size=torch.sum(factor_graph.numFactors).item(),
    )
    factor_beliefs = factor_beliefs + factor_graph.factor_potentials #cal factor beliefs
    factor_beliefs[torch.where(factor_graph.factor_potential_masks==1)] = LN_ZERO
    factor_beliefs = torch.clamp(factor_beliefs, min=LN_ZERO)
    return factor_beliefs
def calculate_beliefs(factor_graph, factorToVar_messages, output_intermedia_results=False):
    var_beliefs = cal_var_beliefs(factor_graph, factorToVar_messages)
    varToFactor_messages = cal_varToFactor_messages(
        factor_graph, var_beliefs, factorToVar_messages,
    )
    expanded_varToFactor_messages  = expand_varToFactor_messages(
        factor_graph, varToFactor_messages
    )
    factor_beliefs = cal_factor_beliefs(
        factor_graph, expanded_varToFactor_messages,
    )
    if output_intermedia_results:
        return var_beliefs, varToFactor_messages, expanded_varToFactor_messages, factor_beliefs
    else:
        return var_beliefs, factor_beliefs
def calibrate_beliefs(beliefs):
    max_beliefs = beliefs
    for dim in range(1, len(beliefs.shape)):
        max_beliefs = torch.max(max_beliefs, dim=dim, keepdims=True)[0]
    beliefs = beliefs - max_beliefs
    beliefs = torch.exp(beliefs)
    sum_beliefs = torch.sum(beliefs, dim=tuple(list(range(1,len(beliefs.shape)))), keepdims=True)
    beliefs = beliefs / sum_beliefs
    return beliefs
class _FixedIterMaxBP(nn.Module):
    def __init__(self, damping=0., normalization_flag=True,
                 maxiter=10000, tol=1e-9,):
        super(_FixedIterMaxBP, self).__init__()
        self.bp_layer = self.generate_bp_layer()
        self.damping, self.normalization_flag = damping, normalization_flag
        self.maxiter, self.tol = maxiter, tol
    def forward_list(self, factor_graph, maxiter=None):
        prv_factorToVar_messages = factor_graph.prv_factorToVar_messages
        prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs = self.bp_layer(
            factor_graph, prv_factorToVar_messages,
            damping=self.damping, normalization_flag=self.normalization_flag
        )
        max_diff = 1.
        maxiter = maxiter or self.maxiter

        var_beliefs_list, factor_beliefs_list = [], []
        for iter_num in range(maxiter):
            if max_diff > self.tol:
                prv_factorToVar_messages, var_beliefs, factor_beliefs = self.bp_layer(
                    factor_graph, prv_factorToVar_messages,
                    damping=self.damping, normalization_flag=self.normalization_flag
                )
                max_diff = max(
                    torch.max(torch.abs(calibrate_beliefs(factor_beliefs)-calibrate_beliefs(prv_factor_beliefs))).item(),
                    torch.max(torch.abs(calibrate_beliefs(var_beliefs)-calibrate_beliefs(prv_var_beliefs))).item(),
                )
                prv_var_beliefs, prv_factor_beliefs = var_beliefs, factor_beliefs

                var_beliefs_list.append(var_beliefs)
                var_beliefs_list[-1][factor_graph.var_beliefs_masks] = -np.inf
                var_beliefs_list[-1] = calibrate_beliefs(var_beliefs_list[-1])
                factor_beliefs_list.append(factor_beliefs)
                factor_beliefs_list[-1][factor_graph.factor_beliefs_masks] = -np.inf
                factor_beliefs_list[-1] = calibrate_beliefs(factor_beliefs_list[-1])
        return var_beliefs_list, factor_beliefs_list
    def forward(self, factor_graph):
        var_beliefs_list, factor_beliefs_list = self.forward_list(factor_graph)
        return var_beliefs_list[-1], factor_beliefs_list[-1]
    def generate_bp_layer(self,):
        raise NotImplementedError

class MaxBPLayer(nn.Module):
    '''
    Belief Propagation imitating the libdai implementation
    '''
    def __init__(self, mode='map'):
        super(MaxBPLayer, self).__init__()
        if mode not in ["marginal", "map"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.mode = mode
    def cal_new_messages(
        self, factor_graph, prv_factorToVar_messages,
        normalization_flag, damping,
        prv_var_beliefs, prv_varToFactor_messages,
        expanded_prv_varToFactor_messages, prv_factor_beliefs,
    ):
        mapped_prv_factor_beliefs = prv_factor_beliefs[factor_graph.facToVar_edge_idx[0]] #map node beliefs to edges
        new_factorToVar_messages = mapped_prv_factor_beliefs-expanded_prv_varToFactor_messages #avoid double counting
        new_factorToVar_messages = new_factorToVar_messages - torch.max(new_factorToVar_messages)
        if self.mode=='map':
            new_factorToVar_messages = scatter_max(
                src=new_factorToVar_messages.reshape([new_factorToVar_messages.numel()])[factor_graph.facStates_to_varIdx>=0],
                index=factor_graph.facStates_to_varIdx[factor_graph.facStates_to_varIdx>=0],
                dim_size=prv_factorToVar_messages.size(0)*factor_graph.var_cardinality[0].item() + 1
            )[0]
        else:
            new_factorToVar_messages = scatter_sum(
                src=new_factorToVar_messages.reshape([new_factorToVar_messages.numel()])[factor_graph.facStates_to_varIdx>=0],
                index=factor_graph.facStates_to_varIdx[factor_graph.facStates_to_varIdx>=0],
                dim_size=prv_factorToVar_messages.size(0)*factor_graph.var_cardinality[0].item() + 1
            )
        new_factorToVar_messages = new_factorToVar_messages[:-1].reshape(
            prv_factorToVar_messages.shape
        )
        if normalization_flag:
            new_factorToVar_messages = new_factorToVar_messages - logsumexp_multipleDim(new_factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
        new_factorToVar_messages[torch.where(factor_graph.factorToVar_messages_masks)] = LN_ZERO
        new_factorToVar_messages = torch.clamp(new_factorToVar_messages, min=LN_ZERO)
        return new_factorToVar_messages
    def update_messages(
        self, factor_graph, normalization_flag, damping,
        prv_factorToVar_messages, prv_var_beliefs,
        prv_varToFactor_messages, prv_factor_beliefs,
        new_factorToVar_messages, var_beliefs,
        varToFactor_messages, factor_beliefs,
    ):
        factorToVar_messages = damping * prv_factorToVar_messages + (1.-damping) * new_factorToVar_messages
        factorToVar_messages[torch.where(factor_graph.factorToVar_messages_masks)] = LN_ZERO
        factorToVar_messages = torch.clamp(factorToVar_messages, min=LN_ZERO)
        return factorToVar_messages
    def forward(
        self, factor_graph, prv_factorToVar_messages,
        normalization_flag=True, damping=0.,
    ):
        prv_var_beliefs, prv_varToFactor_messages, \
            expanded_prv_varToFactor_messages, prv_factor_beliefs = \
            calculate_beliefs(factor_graph, prv_factorToVar_messages, output_intermedia_results=True)

        new_factorToVar_messages =  self.cal_new_messages(
            factor_graph, prv_factorToVar_messages,
            normalization_flag, damping,
            prv_var_beliefs, prv_varToFactor_messages,
            expanded_prv_varToFactor_messages, prv_factor_beliefs
        )

        var_beliefs, varToFactor_messages, _, factor_beliefs = calculate_beliefs(
            factor_graph, new_factorToVar_messages, output_intermedia_results=True
        )

        factorToVar_messages = self.update_messages(
            factor_graph, normalization_flag, damping,
            prv_factorToVar_messages, prv_var_beliefs,
            prv_varToFactor_messages, prv_factor_beliefs,
            new_factorToVar_messages, var_beliefs,
            varToFactor_messages, factor_beliefs,
        )

        return factorToVar_messages, var_beliefs, factor_beliefs
class MaxBP(_FixedIterMaxBP):
    def __init__(self, damping=0., normalization_flag=True, maxiter=10000, tol=1e-9):
        super(MaxBP, self).__init__(
            damping=damping, normalization_flag=normalization_flag,
            maxiter=maxiter, tol=tol,
        )
    def generate_bp_layer(self,):
        return MaxBPLayer()

class ScheduleMaxBPLayer(MaxBPLayer):
    def __init__(self, src_names=['diff_f2v_messages', 'diff_beliefs'],
                 hidden_size=64, batch_norm_flag=True, damping=0.):
        super(ScheduleMaxBPLayer, self).__init__()
        self.src_names = src_names

        input_channel_list = {
            'diff_f2v_messages': 3,
            'diff_beliefs': 2,
        }
        input_channel = np.sum([input_channel_list[sn] for sn in src_names])

        self.update_distance_module = nn.Sequential(
            nn.Linear(input_channel, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size) if batch_norm_flag else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size) if batch_norm_flag else nn.Identity(),
            nn.LeakyReLU(),
        )
        self.update_damping_module = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
        damping = min(max(damping, LN_ZERO), 1-LN_ZERO)
        inverse_damping = np.log(damping/(1-damping))
        self.update_damping_module[0].weight = nn.Parameter(torch.zeros_like(self.update_damping_module[0].weight))
        self.update_damping_module[0].bias = nn.Parameter(inverse_damping+torch.zeros_like(self.update_damping_module[0].bias))
    def update_messages(
        self, factor_graph, normalization_flag, damping,
        prv_factorToVar_messages, prv_var_beliefs,
        prv_varToFactor_messages, prv_factor_beliefs,
        new_factorToVar_messages, var_beliefs,
        varToFactor_messages, factor_beliefs,
    ):
        input_feat_list = []
        if 'diff_f2v_messages' in self.src_names:
            diff_factorToVar_messages = torch.abs(prv_factorToVar_messages - new_factorToVar_messages)
            normed_new_factorToVar_messages = new_factorToVar_messages - logsumexp_multipleDim(new_factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
            normed_prv_factorToVar_messages = prv_factorToVar_messages - logsumexp_multipleDim(prv_factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
            diff_normed_factorToVar_messages = torch.abs(normed_prv_factorToVar_messages - normed_new_factorToVar_messages)
            diff_exp_normed_factorToVar_messages = torch.abs(
                torch.exp(normed_prv_factorToVar_messages) - torch.exp(normed_new_factorToVar_messages)
            )
            input_feat_list += [diff_factorToVar_messages, diff_normed_factorToVar_messages, diff_exp_normed_factorToVar_messages]
        if 'diff_beliefs' in self.src_names:
            mapped_factor_beliefs = factor_beliefs[factor_graph.facToVar_edge_idx[0]]
            maximized_factor_beliefs = scatter_max(
                src=mapped_factor_beliefs.reshape([mapped_factor_beliefs.numel()])[factor_graph.facStates_to_varIdx>=0],
                index=factor_graph.facStates_to_varIdx[factor_graph.facStates_to_varIdx>=0],
                dim_size=prv_factorToVar_messages.size(0)*factor_graph.var_cardinality[0].item() + 1
            )[0]
            maximized_factor_beliefs = maximized_factor_beliefs[:-1].reshape(
                prv_factorToVar_messages.shape
            )
            maximized_factor_beliefs = maximized_factor_beliefs - logsumexp_multipleDim(maximized_factor_beliefs, dim=0).unsqueeze(-1)
            normed_var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim=0).unsqueeze(-1)
            mapped_normed_var_beliefs = normed_var_beliefs[factor_graph.facToVar_edge_idx[1]]
            diff_beliefs = torch.abs(maximized_factor_beliefs - mapped_normed_var_beliefs)
            diff_exp_beliefs = torch.abs(torch.exp(maximized_factor_beliefs) - torch.exp(mapped_normed_var_beliefs))
            input_feat_list += [diff_beliefs, diff_exp_beliefs]

        input_feat = torch.stack(input_feat_list, dim=-1)
        numMsgs, var_card, feat_dim = input_feat.shape
        distance = self.update_distance_module(input_feat.reshape([-1, feat_dim]))
        distance = distance.reshape([numMsgs, var_card, -1])
        _expanded_f2v_messages_masks = factor_graph.factorToVar_messages_masks.unsqueeze(-1).expand_as(distance)
        distance[torch.where(_expanded_f2v_messages_masks)] = LN_ZERO
        distance = torch.max(distance, dim=1)[0]
        damping = self.update_damping_module(distance)

        factorToVar_messages = damping * prv_factorToVar_messages + (1.-damping) * new_factorToVar_messages
        factorToVar_messages[torch.where(factor_graph.factorToVar_messages_masks)] = LN_ZERO
        factorToVar_messages = torch.clamp(factorToVar_messages, min=LN_ZERO)
        return factorToVar_messages
class ScheduleMaxBP(_FixedIterMaxBP):
    def __init__(self, damping=0., normalization_flag=True, maxiter=10000, tol=1e-9,
                 src_names=['diff_f2v_messages', 'diff_beliefs'],
                 hidden_size=64, batch_norm_flag=True,):
        self.src_names = src_names
        self.hidden_size = hidden_size
        self.batch_norm_flag = batch_norm_flag
        self.damping = damping
        super(ScheduleMaxBP, self).__init__(
            damping=damping, normalization_flag=normalization_flag,
            maxiter=maxiter, tol=tol,
        )
    def generate_bp_layer(self,):
        return ScheduleMaxBPLayer(
            src_names=self.src_names, hidden_size=self.hidden_size,
            batch_norm_flag=self.batch_norm_flag,
            damping = self.damping,
        )

class GBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(GBatchNorm1d, self).__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=False,
        )
    def forward(self, x, batch, num_graphs=None):
        cur_mean = scatter_mean(x, batch, dim=0, dim_size=num_graphs)
        assert(not torch.isnan(cur_mean).any())
        assert(not torch.isinf(cur_mean).any())
        cur_mean_square = scatter_mean(x*x, batch, dim=0, dim_size=num_graphs)
        assert(not torch.isnan(cur_mean_square).any())
        assert(not torch.isinf(cur_mean_square).any())
        cur_var = torch.clamp(cur_mean_square - cur_mean*cur_mean, min=0)
        assert(not torch.isnan(cur_var).any())
        assert(not torch.isinf(cur_var).any())
        normed_x = (x-cur_mean[batch])/(torch.sqrt(cur_var+self.eps*self.eps)[batch])
        var_mask = (cur_var < self.eps).float()[batch]
        normed_x = (1-var_mask)*normed_x + var_mask*(x-cur_mean[batch])
        assert(not torch.isnan(normed_x).any()), (self.eps, torch.sum(torch.isnan(x)), torch.sum(torch.isnan(cur_mean)), torch.sum(torch.isnan(cur_var)), cur_var[torch.isnan(torch.sqrt(cur_var))], torch.sum(torch.isnan(torch.sqrt(cur_var))), torch.sum(torch.isnan(normed_x)))
        assert(not torch.isinf(normed_x).any())
        if self.affine:
            normed_x = normed_x*self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        assert(not torch.isnan(normed_x).any())
        assert(not torch.isinf(normed_x).any())
        return normed_x

class ScheduleMaxBPLayerV2(MaxBPLayer):
    def __init__(self, mode='map',
                 src_names=['diff_f2v_messages', 'diff_beliefs'],
                 hidden_size=64, batch_norm_flag=True, damping=0.):
        super(ScheduleMaxBPLayerV2, self).__init__(mode=mode)
        self.src_names = src_names
        self.batch_norm_flag = batch_norm_flag

        input_channel_list = {
            'diff_f2v_messages': 2,
            'diff_beliefs': 1,
        }
        input_channel = np.sum([input_channel_list[sn] for sn in src_names])

        self.update_distance_linear1 = nn.Linear(input_channel, hidden_size, bias=True)
        self.update_distance_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.batch_norm_flag:
            self.update_distance_norm1 = GBatchNorm1d(hidden_size)
            self.update_distance_norm2 = GBatchNorm1d(hidden_size)
        self.update_damping_module = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
        damping = min(max(damping, LN_ZERO), 1-LN_ZERO)
        inverse_damping = np.log(damping/(1-damping))
        self.update_damping_module[0].weight = nn.Parameter(torch.zeros_like(self.update_damping_module[0].weight))
        self.update_damping_module[0].bias = nn.Parameter(inverse_damping+torch.zeros_like(self.update_damping_module[0].bias))
    def update_messages(
        self, factor_graph, normalization_flag, damping,
        prv_factorToVar_messages, prv_var_beliefs,
        prv_varToFactor_messages, prv_factor_beliefs,
        new_factorToVar_messages, var_beliefs,
        varToFactor_messages, factor_beliefs,
    ):
        input_feat_list = []
        if 'diff_f2v_messages' in self.src_names:
            diff_factorToVar_messages = torch.abs(prv_factorToVar_messages - new_factorToVar_messages)
            normed_new_factorToVar_messages = new_factorToVar_messages - logsumexp_multipleDim(new_factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
            normed_prv_factorToVar_messages = prv_factorToVar_messages - logsumexp_multipleDim(prv_factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
            diff_normed_factorToVar_messages = torch.abs(normed_prv_factorToVar_messages - normed_new_factorToVar_messages)
            # diff_exp_normed_factorToVar_messages = torch.abs(
                # torch.exp(normed_prv_factorToVar_messages) - torch.exp(normed_new_factorToVar_messages)
            # )
            # input_feat_list += [diff_factorToVar_messages, diff_normed_factorToVar_messages, diff_exp_normed_factorToVar_messages]
            input_feat_list += [diff_factorToVar_messages, diff_normed_factorToVar_messages, ]
        if 'diff_beliefs' in self.src_names:
            mapped_factor_beliefs = factor_beliefs[factor_graph.facToVar_edge_idx[0]]
            maximized_factor_beliefs = scatter_max(
                src=mapped_factor_beliefs.reshape([mapped_factor_beliefs.numel()])[factor_graph.facStates_to_varIdx>=0],
                index=factor_graph.facStates_to_varIdx[factor_graph.facStates_to_varIdx>=0],
                dim_size=prv_factorToVar_messages.size(0)*factor_graph.var_cardinality[0].item() + 1
            )[0]
            maximized_factor_beliefs = maximized_factor_beliefs[:-1].reshape(
                prv_factorToVar_messages.shape
            )
            maximized_factor_beliefs = maximized_factor_beliefs - logsumexp_multipleDim(maximized_factor_beliefs, dim=0).unsqueeze(-1)
            normed_var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim=0).unsqueeze(-1)
            mapped_normed_var_beliefs = normed_var_beliefs[factor_graph.facToVar_edge_idx[1]]
            diff_beliefs = torch.abs(maximized_factor_beliefs - mapped_normed_var_beliefs)
            # diff_exp_beliefs = torch.abs(torch.exp(maximized_factor_beliefs) - torch.exp(mapped_normed_var_beliefs))
            # input_feat_list += [diff_beliefs, diff_exp_beliefs]
            input_feat_list += [diff_beliefs, ]

        input_feat = torch.stack(input_feat_list, dim=-1)
        numMsgs, var_card, feat_dim = input_feat.shape
        tmp_input_feat = input_feat.reshape([-1, feat_dim])
        distance = self.update_distance_linear1(tmp_input_feat)
        if self.batch_norm_flag:
            batch = torch.cat([vi+torch.zeros([vn], dtype=torch.long, device=input_feat.device) for vi,vn in enumerate(factor_graph.numVars)], dim=0)
            batch = batch[factor_graph.facToVar_edge_idx[1]]
            batch = batch.unsqueeze(-1).expand([numMsgs, var_card]).reshape([-1])
            distance = self.update_distance_norm1(distance, batch, num_graphs=factor_graph.numVars.size(0))
        distance = F.leaky_relu(distance)
        distance = self.update_distance_linear2(distance)
        if self.batch_norm_flag:
            distance = self.update_distance_norm2(distance, batch, num_graphs=factor_graph.numVars.size(0))
        distance = F.leaky_relu(distance)

        distance = distance.reshape([numMsgs, var_card, -1])
        _expanded_f2v_messages_masks = factor_graph.factorToVar_messages_masks.unsqueeze(-1).expand_as(distance)
        distance[torch.where(_expanded_f2v_messages_masks)] = LN_ZERO
        distance = torch.max(distance, dim=1)[0]
        damping = self.update_damping_module(distance)

        factorToVar_messages = damping * prv_factorToVar_messages + (1.-damping) * new_factorToVar_messages
        factorToVar_messages[torch.where(factor_graph.factorToVar_messages_masks)] = LN_ZERO
        factorToVar_messages = torch.clamp(factorToVar_messages, min=LN_ZERO)
        return factorToVar_messages
class ScheduleMaxBPV2(_FixedIterMaxBP):
    def __init__(self, mode='map',
                 damping=0., normalization_flag=True, maxiter=10000, tol=1e-9,
                 src_names=['diff_f2v_messages', 'diff_beliefs'],
                 hidden_size=64, batch_norm_flag=True,):
        if mode not in ["marginal", "map"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.mode = mode
        self.src_names = src_names
        self.hidden_size = hidden_size
        self.batch_norm_flag = batch_norm_flag
        self.damping = damping
        super(ScheduleMaxBPV2, self).__init__(
            damping=damping, normalization_flag=normalization_flag,
            maxiter=maxiter, tol=tol,
        )
    def generate_bp_layer(self,):
        return ScheduleMaxBPLayerV2(
            src_names=self.src_names, hidden_size=self.hidden_size,
            batch_norm_flag=self.batch_norm_flag,
            damping = self.damping, mode=self.mode,
        )

