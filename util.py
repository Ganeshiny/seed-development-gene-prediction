import csv
import glob
import numpy as np
from sklearn.metrics import average_precision_score
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import torch
import numpy as np


# Function to one-hot encode amino acid sequences
def one_hot_encode_sequence(sequence, aa_list):
    mapping = {aa: i for i, aa in enumerate(aa_list)}
    one_hot = np.zeros((len(sequence), len(aa_list)))
    for i, aa in enumerate(sequence):
        if aa in mapping:
            one_hot[i, mapping[aa]] = 1
        else:
            one_hot[i, -1] = 1  # For unknown amino acids
    return one_hot

def pdb_to_distmatrix(pdbFile):
    # Getting the residues from the pdb file
    parser = PDBParser()
    structure = parser.get_structure(pdbFile.split('/')[-1].split('.')[0], pdbFile)
    residues = [r for r in structure.get_residues()]

    dist_matrix = np.zeros((len(residues), len(residues)))
    for row_n,residue_one in enumerate(residues):
        for col_n, residue_two in enumerate(residues):
            #residue distance
            dist_matrix[row_n, col_n]= np.array(np.linalg.norm([residue_one["CA"].get_coord()- residue_two["CA"].get_coord()]))

    return dist_matrix
  
pdb_to_distmatrix("C:/Users/LENOVO/Desktop/seed-development-gene-prediction/1S3P-A.pdb")
#load_predicted_PDB("C:/Users/LENOVO/Desktop/seed-development-gene-prediction/1S3P-A.pdb")