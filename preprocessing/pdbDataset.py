import csv
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing

#DO THE MULTIPROCESSING HERE!


'''
This is a fucntion to one hot encode the sequence

'''

def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

'''
Loads the nrPDB-GO-train - list of sequences with the pdb id and the chain index
'''
def load_list(fname):
    pdb_chain_list = []
    with open(fname, 'r') as fRead:
        for line in fRead:
            pdb_chain_list.append(line.strip())
    return pdb_chain_list


'''
Loads the GO annot file with the .tsv extension
'''

def load_GO_annot(filename):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                  goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
    return prot2annot, goterms, gonames


'''
Dataset class using pytorch
'''
class MyDataset(Dataset):
    def __init__(self, prot_list, prot2annot, npz_dir):
        self.prot_list = prot_list 
        self.prot2annot = prot2annot 
        self.npz_dir = npz_dir
        self.num_shards = num_shards
        self.indices = indices

    def __len__(self):
        return len(self.prot_list) #The number of IDs we are using to train/test

    def _serialize_example(self, prot_id, sequence, ca_dist_matrix, cb_dist_matrix):
        #print(self.prot2annot[0][prot_id])
        labels = torch.from_numpy(self.prot2annot[0][prot_id]['molecular_function']).long()
        labels = torch.from_numpy(self.prot2annot[0][prot_id]['biological_process']).long()
        labels = torch.from_numpy(self.prot2annot[0][prot_id]['cellular_component']).long()
        d_feature = {
            'prot_id': prot_id.encode(),
            'seq_1hot': torch.from_numpy(seq2onehot(sequence).reshape(-1)).float(),
            'L': torch.tensor(len(sequence)),
        }
        
        d_feature['mf_labels'] = labels[:len(labels)//3]
        d_feature['bp_labels'] = labels[len(labels)//3:2*(len(labels)//3)]
        d_feature['cc_labels'] = labels[2*(len(labels)//3):]

        d_feature['ca_dist_matrix'] = torch.from_numpy(ca_dist_matrix.reshape(-1)).float()
        d_feature['cb_dist_matrix'] = torch.from_numpy(cb_dist_matrix.reshape(-1)).float()
        return d_feature

    def __getitem__(self, idx):
        prot = self.prot_list[idx]
        #print(self.prot_list)
        pdb_file = os.path.normpath(os.path.join(self.npz_dir, f'{prot}.npz'))
        if os.path.isfile(pdb_file):
            cmap = np.load(pdb_file)
            sequence = str(cmap['seqres'])
            ca_dist_matrix = cmap['C_alpha']
            cb_dist_matrix = cmap['C_beta']
            #print(f"The dict im trying to print: {self.prot2annot}")
            
            example = self._serialize_example(prot, sequence, ca_dist_matrix, cb_dist_matrix)
            return example
        else:
            print(pdb_file)
            return None

def serialize_dataset(args, idx):
    pt_file = f'{args.serial_prefix}_{idx:0>2d}-of-{args.num_shards:0>2d}.pt'
    print(f"### Serializing {len(args.prot_list)} examples into {pt_file}")

    tmp_prot_list = args.prot_list[args.indices[idx][0]:args.indices[idx][1]]
    dataset = MyDataset(tmp_prot_list, load_GO_annot(args.annot), args.npz_dir)


    torch.save(dataset, pt_file)
    print(f"Writing {pt_file} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='./data/nrPDB-GO_annot.tsv',
                        help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-prot_list', type=str, default='./data/nrPDB-GO_train.txt',
                        help="Input file (*.txt) with a set of protein IDs with distMAps in npz_dir.")
    parser.add_argument('-npz_dir', type=str, default='./data/annot_pdb_chains_npz')
    parser.add_argument('-serial_prefix', type=str, default= 'serialized_data')
    parser.add_argument('-num_shards', type=int, default=20, help="Number of pytorch files per protein set.")

    args = parser.parse_args()

    # Load protein list and annotations
    prot_list = load_list(args.prot_list)
    prot2annot, _, _ = load_GO_annot(args.annot)


    # Number of shards for serialization
    num_shards = 20
    shard_size = len(prot_list) // num_shards
    indices = [(i * shard_size, (i + 1) * shard_size) for i in range(0, num_shards)]
    indices[-1] = (indices[-1][0], len(prot_list))

    # Save indices for later use
    args.indices = indices
    args.prot_list = prot_list

    # Serialize datasets
    for idx in range(num_shards):
        serialize_dataset(args, idx)
