'''
This script consists of a PDB Batch Processor class that processes a directory of legacy PDB files.

- get_CA_coordinates: Returns a numpy vector of coordinates of C-alpha (CA) distances of each residue in the protein.
- create_distance_matrix: Generates a symmetric matrix of Euclidean distances between the residues (CA atoms) using CA coordinates.
- create_cmap: Creates a contact map (a .npy file) using the distance matrix with a certain threshold.
- visualize_cmap: Visualizes the contact map file.
- extract_PDB_gz: When batch downloading PDB.gz files, they need to be extracted into a target directory before being processed.

'''

'''
Problems I'm facing here:

A single PDB file has several chains, I can have them all in a single contact map or can have in different contact maps
These maps are of different sizes. What happens if i pad them?

'''

import os
import gzip
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser


class PDB_Batch_Processor:
    def __init__(self, PDB_file_path, threshold_distance=10.0):
        '''
        initiating this class using the path containing the extracted pdb files that needs to be processed

        PDB_Batch_Processor class attributes

        ::PDB_id:: extracted PDB ID from the file path
        ::structure:: structure class using the PDB parser module
        ::threshold_distance:: stores the minimum distance threshold between two residues
        '''
        self.PDB_id = os.path.splitext(os.path.basename(PDB_file_path))[0]
        self.structure = PDBParser(QUIET=True).get_structure(self.PDB_id, PDB_file_path)
        self.threshold_distance = threshold_distance
    
    @staticmethod
    def extract_PDB_gz(PDB_gz_directory, output_PDB_directory):
        '''
        extracts the pdb.gz files to the target directory which is used to run the rest of the funtions
        '''
        PDB_gz_files = [f for f in os.listdir(PDB_gz_directory) if f.endswith('.pdb.gz')]
        extracted_PDB_files = []

        for PDB_gz_file in PDB_gz_files:
            PDB_id = os.path.splitext(PDB_gz_file)[0]
            with gzip.open(os.path.join(PDB_gz_directory, PDB_gz_file), 'rb') as f:
                uncompressed_content = f.read()

            output_PDB_path = os.path.join(output_PDB_directory, f"{PDB_id}.pdb") # Target directory to store all the extracted PDB files
            with open(output_PDB_path, 'wb') as new_PDB_file:
                new_PDB_file.write(uncompressed_content)

            extracted_PDB_files.append(output_PDB_path)

        return extracted_PDB_files

    def get_CA_coordinates(self, residues, num_residues):
        '''
        Gets the CA distances between the residues
        Note: some of the pdb files are missing the coordinates for the CA distance
        '''
        avg_distance = 10.0  # np.mean(self.dist_matrix)
        coords = {}
        for i, residue in enumerate(residues):
            try:
                ca_atom = residue['CA']
                coords[i] = ca_atom.get_coord()
            except KeyError:
                coords[i] = np.full(3, avg_distance)
        return coords

    def create_distance_matrix(self, coords, num_residues):
        '''
        Creates distance matrices for a single pdb file using the CA coordinates from get_CA_coordinates
        '''
        dist_matrix = np.zeros((num_residues, num_residues))  # creating a symmetric matrix of the size of the length of the protein

        for i in range(num_residues):
            for j in range(num_residues):
                if i in coords and j in coords:
                    dist = np.linalg.norm(coords[i] - coords[j]) if i != j else 0  # Set diagonal elements to 0
                else:
                    dist = 10.0  # Assign some default value in case the coordinate is missing
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def create_cmap(self, dist_matrix):
        '''
        Create adjacency matrices for residues situated closer than the threshold distance.
        When residues are at a distance less than the threshold distance (ideally 6-12 Angstroms),
        the matrix element is assigned 1, otherwise 0.
        '''
        contact_map = dist_matrix <= self.threshold_distance
        return contact_map

    def create_contact_maps_for_chains(self, cmap_directory):

        '''
        Create contact maps for each chain in the PDB structure.

        Processes each chain in the PDB structure and generates a contact map for each chain.
        The contact map is saved as a .npy file in the specified directory.
        '''
                
        os.makedirs(cmap_directory, exist_ok=True)
        
        for model in self.structure:
            for chain in model:
                chain_id = chain.id
                residues = list(chain.get_residues())
                coords = self.get_CA_coordinates(residues, len(residues))
                dist_matrix = self.create_distance_matrix(coords, len(residues))
                contact_map = self.create_cmap(dist_matrix)
                
                # Save the contact map for the current chain
                filename = f"{self.PDB_id}_chain_{chain_id}_contact_map.npy"
                filepath = os.path.join(cmap_directory, filename)
                np.save(filepath, contact_map)

    def visualize_cmap(self, cmap_directory, chain_id=None, ):
        '''
        Visualizes the contact map for a specified chain

        Loads and displays the contact map for the specified chain.
        '''
        if chain_id:
            cmap_filename = f"{self.PDB_id}_chain_{chain_id}_contact_map.npy"
            cmap_path = os.path.join(cmap_directory, cmap_filename)
            contact_map = np.load(cmap_path)
            plt.imshow(contact_map, cmap='viridis', interpolation='none')
            plt.colorbar()
            plt.title(f'Contact Map for {self.PDB_id} Chain {chain_id}')
            plt.xlabel('Residue Index')
            plt.ylabel('Residue Index')
            plt.show()



PDB_gz_directory = "../seed-development-gene-prediction/pdb_preprocessing/pdb_train"
output_PDB_directory = "../seed-development-gene-prediction/pdb_preprocessing/extracted_pdb_legacy_files"
cmap_directory = "../seed-development-gene-prediction/pdb_preprocessing/contact_maps"
extracted_PDB_files = PDB_Batch_Processor.extract_PDB_gz(PDB_gz_directory, output_PDB_directory)

for PDB_file in extracted_PDB_files:
    PDB_batch_processor_instance = PDB_Batch_Processor(PDB_file)
    PDB_batch_processor_instance.create_contact_maps_for_chains(cmap_directory)
    for model in PDB_batch_processor_instance.structure:
        for chain in model:
            chain_id = chain.id
            PDB_batch_processor_instance.visualize_cmap(cmap_directory, chain_id)

    




    
