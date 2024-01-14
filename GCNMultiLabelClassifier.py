import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes_dict):
        super(GCN, self).__init__()
        
        # Initializing GCN layers
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        # Output layers for each ontology, ModuleDict holds submodules in a dictionary.
        # Here the submodules are the Linear output layers for each ontology
        self.output_layers = nn.ModuleDict({
            ontology: nn.Linear(hidden_size, output_size) for ontology, output_size in output_sizes_dict.items()
        })

        # Dropout layers
        self.dropout_input = nn.Dropout(0.6)
        self.dropout_hidden = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout_input(x)

        # Second GCN layer
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout_hidden(x)

        # Aggregation step
        x = global_mean_pool(x, batch)  # or global_add_pool or global_max_pool

        # Output layers for each ontology
        outputs = {ontology: self.output_layers[ontology](x) for ontology in self.output_layers.keys()}
        return outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)