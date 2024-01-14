import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocessing.pydataset3 import PDB_Dataset  # Make sure to import your dataset module
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss
from GCNMultiLabelClassifier import GCN


'''
Add the best parameters and hyper parameters here, and modify the GCN code to use it from here

'''

'''
Loss function 
'''
class CustomMultilabelLoss(nn.Module):
    def __init__(self):
        super(CustomMultilabelLoss, self).__init__()

    def forward(self, predictions, targets):
        # Assuming predictions and targets are dictionaries
        loss = {}

        for ontology, prediction in predictions.items():
            # Assuming BCELoss for each ontology
            # Ensure both prediction and target are 1D tensors
            bce_loss = nn.BCEWithLogitsLoss()(prediction.view(-1), targets[ontology].float().view(-1))
            loss[ontology] = bce_loss

        # Calculate 'total' loss
        loss['total'] = sum(loss.values())

        return loss