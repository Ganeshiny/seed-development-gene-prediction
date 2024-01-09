import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocessing.pydataset3 import PDB_Dataset  # Make sure to import your dataset module
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss

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

class GCNMultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes_dict):
        super(GCNMultiLabelClassifier, self).__init__()
        
        # Initializing GCN layers
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        # Output layers for each ontology, ModuleDict holds submodules in a dictionary.
        # Here the submodules are the Linear output layers for each ontology
        self.output_layers = nn.ModuleDict({
            ontology: nn.Linear(hidden_size, output_size) for ontology, output_size in output_sizes_dict.items()
        })

        # Dropout layers
        self.dropout_input = nn.Dropout(0.8)
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

# Set up the dataset
root = 'preprocessing/data/annot_pdb_chains_npz'
annot_file = 'preprocessing/data/nrPDB-GO_annot.tsv'
num_shards = 20

# Load data using DataLoader directly
dataset = PDB_Dataset(root, annot_file, num_shards=num_shards)
torch.manual_seed(12345)

# Split dataset into train, validation, and test sets
train_dataset, temp_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)
val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=12345)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, criterion, and optimizer
input_size = len(dataset[0].x[0])
hidden_size = 4
output_sizes_dict = {
    'molecular_function': len(dataset[0].y['molecular_function']),
    'biological_process': len(dataset[0].y['biological_process']),
    'cellular_component': len(dataset[0].y['cellular_component'])
}

model = GCNMultiLabelClassifier(input_size, hidden_size, output_sizes_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = CustomMultilabelLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 50
train_losses = []
val_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}
test_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0.0

    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        targets = {ontology: data.y[ontology] for ontology in output_sizes_dict.keys()}
        loss = criterion(outputs, targets)
        loss['total'].backward()
        optimizer.step()
        total_train_loss += loss['total'].item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    all_val_preds = {ontology: [] for ontology in output_sizes_dict.keys()}
    all_val_labels = {ontology: [] for ontology in output_sizes_dict.keys()}

    with torch.no_grad():
        for data in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)

            for ontology in output_sizes_dict.keys():
                all_val_preds[ontology].extend(torch.sigmoid(outputs[ontology].view(-1)).cpu())
                all_val_labels[ontology].extend(data.y[ontology].cpu())

    for ontology in output_sizes_dict.keys():
        val_accuracy = accuracy_score(torch.vstack(all_val_labels[ontology]),
                                       (torch.vstack(all_val_preds[ontology]) > 0.5).int())
        val_accuracies[ontology].append(val_accuracy.item())

    # Testing
    model.eval()
    all_test_preds = {ontology: [] for ontology in output_sizes_dict.keys()}
    all_test_labels = {ontology: [] for ontology in output_sizes_dict.keys()}

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Testing'):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)

            for ontology in output_sizes_dict.keys():
                all_test_preds[ontology].extend(torch.sigmoid(outputs[ontology].view(-1)).cpu())
                all_test_labels[ontology].extend(data.y[ontology].cpu())

    for ontology in output_sizes_dict.keys():
        test_accuracy = accuracy_score(torch.vstack(all_test_labels[ontology]),
                                        (torch.vstack(all_test_preds[ontology]) > 0.5).int())
        test_accuracies[ontology].append(test_accuracy.item())

    print(f'Epoch {epoch + 1}/{num_epochs} - '
          f'Training Loss: {avg_train_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

def plot_metrics(train_losses, val_accuracies, test_accuracies, epochs):
    plt.figure(figsize=(12, 5))

    # Plotting Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plotting Validation Accuracy
    plt.subplot(1, 2, 2)
    for ontology, val_accuracy in val_accuracies.items():
        plt.plot(range(1, epochs + 1), val_accuracy, label=f'Validation {ontology} Accuracy', marker='o')

    for ontology, test_accuracy in test_accuracies.items():
        plt.plot(range(1, epochs + 1), test_accuracy, label=f'Test {ontology} Accuracy', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracies over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage:
plot_metrics(train_losses, val_accuracies, test_accuracies, num_epochs)
