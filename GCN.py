import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from preprocessing.pydataset3 import PDB_Dataset
import torch.nn.functional as F


class CustomMultilabelLoss(nn.Module):
    def __init__(self, lambda_reg=0.001):
        super(CustomMultilabelLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, predictions, targets, model):
        # Assuming predictions and targets are dictionaries
        loss = 0.0

        for ontology, prediction in predictions.items():
            target_size = targets[ontology].size(-1)
            prediction = prediction.view(-1, target_size)
            target = targets[ontology].float().view(-1, target_size)

            # Calculate weighted BCE loss for each sample in the batch
            weighted_bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='sum')

            # L2 regularization on model parameters
            l2_reg = torch.sum(torch.stack([torch.norm(param, p=2) for param in model.parameters()]))

            # Add L2 regularization term to the loss
            total_loss = weighted_bce_loss + self.lambda_reg * l2_reg

            # Accumulate the loss
            loss += total_loss

        # Calculate the average loss over all samples in the batch
        average_loss = loss / len(predictions)

        return average_loss

class GCNMultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes_dict, dropout_rate=0.5):
        super(GCNMultiLabelClassifier, self).__init__()

        # Define GCN layers
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        # Output layers for each ontology
        self.output_layers = nn.ModuleDict({
            ontology: nn.Linear(hidden_size, output_size) for ontology, output_size in output_sizes_dict.items()
        })

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        # Second GCN layer
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # Aggregation step (choose one of the following)
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

# Initialize the model
input_size = len(dataset[0].x[0])
hidden_size = 64
output_sizes_dict = {'molecular_function': len(dataset[0].y['molecular_function']),
                     'biological_process': len(dataset[0].y['biological_process']),
                     'cellular_component': len(dataset[0].y['cellular_component'])}

model = GCNMultiLabelClassifier(input_size, hidden_size, output_sizes_dict)

# Choose device (cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up optimizer, scheduler, and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = CustomMultilabelLoss()

# Training loop
num_epochs = 10
train_losses = []
val_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}
test_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        output_flattened = {ontology: torch.flatten(output[ontology]).to(device) for ontology in output.keys()}
        print(f"output: {output_flattened['molecular_function'].shape}") #torch.Size([64, 981])

        # Assuming data.y is a dictionary of labels
        targets = {ontology: data.y[ontology].to(device) for ontology in output.keys()}
        print(f"targets: {targets['molecular_function'].shape}") #torch.Size([62784])

        loss = criterion(output_flattened, targets, model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    y_true_val, y_pred_val = {ontology: [] for ontology in output.keys()}, {ontology: [] for ontology in output.keys()}

    with torch.no_grad():
        for data in tqdm(val_loader, desc=f'Epoch {epoch + 1} - Validating'):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            output_flattened = {ontology: torch.flatten(output[ontology]).to(device) for ontology in output.keys()}

            targets = {ontology: data.y[ontology].unsqueeze(0).to(device) for ontology in output.keys()}
            predictions = {ontology: (output_flattened[ontology] > 0.5).cpu().numpy().flatten() for ontology in output.keys()}

            for ontology in output_flattened.keys():
                y_true_val[ontology].extend(targets[ontology].squeeze().cpu().numpy().flatten())
                y_pred_val[ontology].extend(predictions[ontology])

        # Calculate accuracy on validation set
        accuracy_val = {ontology: accuracy_score(y_true_val[ontology], y_pred_val[ontology]) for ontology in y_true_val.keys()}
        for ontology in val_accuracies.keys():
            val_accuracies[ontology].append(accuracy_val[ontology])

        # Print the calculated accuracies on the validation set
        print("Validation Accuracies:")
        for ontology, acc in accuracy_val.items():
            print(f"{ontology}: {acc}")

    # Testing
    y_true_test, y_pred_test = {ontology: [] for ontology in output.keys()}, {ontology: [] for ontology in output.keys()}

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'Epoch {epoch + 1} - Testing'):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)

            targets = {ontology: data.y[ontology].unsqueeze(0).to(device) for ontology in output.keys()}
            predictions = {ontology: (output[ontology] > 0.5).cpu().numpy().flatten() for ontology in output.keys()}

            for ontology in output.keys():
                y_true_test[ontology].extend(targets[ontology].squeeze().cpu().numpy().flatten())
                y_pred_test[ontology].extend(predictions[ontology])

        # Calculate accuracy on test set
        accuracy_test = {ontology: accuracy_score(y_true_test[ontology], y_pred_test[ontology]) for ontology in y_true_test.keys()}
        for ontology in test_accuracies.keys():
            test_accuracies[ontology].append(accuracy_test[ontology])

        # Print the calculated accuracies on the test set
        print("Test Accuracies:")
        for ontology, acc in accuracy_test.items():
            print(f"{ontology}: {acc}")

    # Adjust the learning rate
    scheduler.step()

# Plotting
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(18, 6))

# Plotting Training Loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Plotting Validation Accuracies
plt.subplot(1, 3, 2)
for ontology in val_accuracies.keys():
    plt.plot(epochs, val_accuracies[ontology], label=f'Validation Accuracy ({ontology})')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()

# Plotting Test Accuracies
plt.subplot(1, 3, 3)
for ontology in test_accuracies.keys():
    plt.plot(epochs, test_accuracies[ontology], label=f'Test Accuracy ({ontology})')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
