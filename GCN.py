import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preprocessing.pydataset3 import PDB_Dataset, pdb_dataset
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class GCNGraphLevel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_output_classes_per_ontology):
        super(GCNGraphLevel, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)
        self.num_output_classes_per_ontology = num_output_classes_per_ontology

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Graph-level pooling (mean pooling)
        x = global_mean_pool(x, batch)

        output = {}
        for ontology in self.num_output_classes_per_ontology.keys():
            output[ontology] = F.log_softmax(x, dim=1)

        return output


root = 'preprocessing/data/annot_pdb_chains_npz'
annot_file = 'preprocessing/data/nrPDB-GO_annot.tsv'
pdb_dataset = PDB_Dataset(root, annot_file)


torch.manual_seed(12345)
dataset = pdb_dataset.shuffle()

train_dataset = pdb_dataset[:int(0.8 * len(pdb_dataset))]
test_dataset = pdb_dataset[int(0.8 * len(pdb_dataset)):]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Initialize the model with updated output dimension
num_output_classes_per_ontology = {ontology: len(classes) for ontology, classes in train_dataset[0].y.items()}
model = GCNGraphLevel(in_channels=26, hidden_channels=64, out_channels=3, num_output_classes_per_ontology=num_output_classes_per_ontology)

print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up optimizer and scheduler (for learning rate scheduling)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_losses = []
train_accuracies = []
test_accuracies = []

num_epochs = 10
best_test_accuracy = 0.0  # To track the best test accuracy
best_epoch = 0

def test(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            for ontology, prediction in output.items():
                target = data.y[ontology].view(-1)
                target = target[:prediction.shape[0]]
                y_true.append(target.cpu().numpy())
                y_pred.append(prediction.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        losses = []
        for ontology, prediction in output.items():
            label_key = ontology
            target = data.y[label_key].view(-1)
            target = target[:prediction.shape[0]]
            loss = F.nll_loss(prediction, target)
            losses.append(loss)

        loss = sum(losses)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            output = model(data)
            for ontology, prediction in output.items():
                target = data.y[ontology].view(-1)
                target = target[:prediction.shape[0]]
                y_true.append(target.cpu().numpy())
                y_pred.append(prediction.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    train_accuracy = accuracy_score(y_true, y_pred)
    train_accuracies.append(train_accuracy)
    train_losses.append(total_loss)

    test_accuracy = test(model, test_loader, device)
    test_accuracies.append(test_accuracy)

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_epoch = epoch
        # Save the best model weights
        torch.save(model.state_dict(), f'best_model_weights.pth')

    print(f'Epoch: {epoch + 1}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Plotting
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Best Test Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch + 1}')
