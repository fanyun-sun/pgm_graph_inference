"""

Convenient constants

"""

import torch

DFLT_DATA_DIR = "./graphical_models/datasets/"
DFLT_MODEL_DIR = "./inference/pretrained"
USE_SPARSE_GNN = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

ASYMMETRIC = False
use_tree_bp = False
use_my_bp = ASYMMETRIC

n_hidden_states = 5
message_dim_P = 2
hidden_unit_message_dim = 64
hidden_unit_readout_dim = 64
T = 10
learning_rate = 1e-2
