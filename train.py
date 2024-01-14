import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from preprocessing.pydataset3 import PDB_Dataset
from gcn-multilabel-classifier import GCN
from config import CustomMultilabelLoss
    
# Set up the dataset
root = 'preprocessing/data/annot_pdb_chains_npz'
annot_file = 'preprocessing/data/nrPDB-GO_annot.tsv'
num_shards = 20

# Load data using DataLoader directly
dataset = PDB_Dataset(root, annot_file, num_shards=num_shards)
torch.manual_seed(12345)

#splitting the dataset into train, test and validation sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.4, random_state=12345)
val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=54321)

#creating dataloader objects out of the train, test and validation datasets
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
print(f"debug: the output_sizes_dict['molecular_function]: {output_sizes_dict['molecular_function']}") #981 : doesn't contain batch size 

#save the best model weights
best_val_accuracy = {ontology: 0.0 for ontology in output_sizes_dict.keys()}
best_model_weights = {ontology: None for ontology in output_sizes_dict.keys()}


#Initiliazing the model
model = GCN(input_size, hidden_size, output_sizes_dict)

#set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss calculation
criterion = CustomMultilabelLoss() 

# Learning rate and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 150
train_losses = []

#accuracies 
train_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}
val_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}
test_accuracies = {ontology: [] for ontology in output_sizes_dict.keys()}

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    all_train_preds = {ontology: [] for ontology in output_sizes_dict.keys()}
    all_train_labels = {ontology: [] for ontology in output_sizes_dict.keys()}

    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        #print(f"debug: outputs['molecular_function]: {outputs['molecular_function'].shape}") #outputs['molecular_function]: torch.Size([64, 981])
        targets = {ontology: data.y[ontology] for ontology in output_sizes_dict.keys()}
        #print(f"debug: targets['molecular_function]: {targets['molecular_function'].shape}") #debug: targets['molecular_function]: torch.Size([62784])
        loss = criterion(outputs, targets) #the outputs[ontology] size is flattened inside the custom loss function here
        loss['total'].backward()
        optimizer.step()
        total_train_loss += loss['total'].item()

        # Calculate training accuracy
        for ontology in output_sizes_dict.keys():
            all_train_preds[ontology].extend(torch.sigmoid(outputs[ontology].view(-1)).cpu())
            all_train_labels[ontology].extend(data.y[ontology].cpu())

    avg_train_loss = total_train_loss / len(train_loader) #/number of batches
    train_losses.append(avg_train_loss)

    for ontology in output_sizes_dict.keys():
        train_accuracy = accuracy_score(torch.vstack(all_train_labels[ontology]),
                                        (torch.vstack(all_train_preds[ontology]) > 0.5).int())
        train_accuracies[ontology].append(train_accuracy.item())

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


        if val_accuracy > best_val_accuracy[ontology]:
            best_val_accuracy[ontology] = val_accuracy
            best_model_weights[ontology] = model.state_dict()

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
          f'Training Accuracy: {train_accuracy:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}')

def plot_metrics(train_losses, train_accuracies, val_accuracies, test_accuracies, epochs):
    plt.figure(figsize=(15, 5))

    # Plotting Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(1, 3, 2)
    for ontology, train_accuracy in train_accuracies.items():
        plt.plot(range(1, epochs + 1), train_accuracy, label=f'Training {ontology} Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracies over Epochs')
    plt.legend()

    # Plotting Validation and Test Accuracies
    plt.subplot(1, 3, 3)
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

# Load the best model weights
for ontology in output_sizes_dict.keys():
    model.load_state_dict(best_model_weights[ontology])
    torch.save(model.state_dict(), f'best_model_weights_{ontology}.pth')

# Usage:
plot_metrics(train_losses, train_accuracies, val_accuracies, test_accuracies, num_epochs)


