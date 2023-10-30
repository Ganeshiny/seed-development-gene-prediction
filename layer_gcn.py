'''
Creating a GraphConv Class
'''
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import Sequential, GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from dataloader import sequences_to_batch

class ProteinGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x) # Might have some changes here
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Initialize the model
input_dim = 26  # Assuming 26 for one-hot encoded amino acid sequences
hidden_dim = 64
output_dim = 10  # Adjust this according to the number of GO annotations

model = ProteinGCN(input_dim, hidden_dim, output_dim)

# Assuming you have training and validation data in the PyTorch Geometric format
# Here X_train is the one-hot encoded amino acid sequences, and y_train is the GO annotations
X_train = sequences_to_batch("C:/Users/LENOVO/Desktop/seed-development-gene-prediction/data/sequences")
y_train = ...

'''
so this is the tricky part
I now have a list of acc and their GOs
I have to split that file into two, one a file with the acc and the other just the GO
After splitting, I need to find all the FASTA files of the acc in the lists, this will be going into the X_train 
the other file must be encoded into a matrix

'''

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(X_train)
    loss = F.nll_loss(out, y_train)  # Assuming the task is a multi-class classification
    loss.backward()
    optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    pred = model(X_val)
    val_loss = F.nll_loss(pred, y_val)  # Calculate loss on validation data
    # Perform validation metrics (accuracy, F1-score, etc.) based on your task




