'''
This code takes in the data and converts to data objects 

'''
import torch
import os
import util
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from util import one_hot_encode_sequence

# Load amino acid sequence data from FASTA file
folder_path = "C:/Users/LENOVO/Desktop/seed-development-gene-prediction/data/sequences"

# Assuming you have a list of amino acids
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

all_sequences = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".fasta") or filename.endswith(".fa"):
        fasta_file = os.path.join(folder_path, filename)
        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
        all_sequences.extend(sequences)

# One-hot encode sequences
one_hot_sequences = [one_hot_encode_sequence(seq, amino_acids) for seq in sequences]

# Convert the one-hot encoded sequences to PyTorch tensors
data = torch.tensor(one_hot_sequences)

# Define the batch size
batch_size = 32

# Create DataLoader for mini-batches
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# Training loop example (using the DataLoader)
for i, batch in enumerate(data_loader):
    # Process the batch
    # Your training process goes here
    print(f"Batch {i + 1}: {batch.shape}")  # Replace with your actual training code
    

'''
how to send my seq from ram to vram 
think of the size youre dealing with 
each seq ebery aa is one hto encoded, repsresented by a vector of lenght 26
and say if a protein i s 100aa long, it has 100 rows and 26 cols
but what dt we use ot store the number
if we use the 64 bit float , thats 64x 2600 full precision
if we use 32 but float its half the prcision
whnen creatng a tensor

'''




