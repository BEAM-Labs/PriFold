from transformers.models.esm.modeling_esm import *
import os
import numpy as np
from torch.utils.data import Dataset
import random
from scipy.ndimage import convolve

class SSDataset(Dataset):
    def __init__(self, df, data_path, tokenizer, aug=None, smooth = None):
        self.df = df
        self.data_path = data_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')
        self.aug = aug
        self.smooth = smooth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, str(file_name) + '.npy')
        ct = np.load(file_path)

        if self.aug:
            seq = self.aug(seq, ct)
        
        if self.smooth:
            smooth_ct = self.label_smoothing(ct)
            return seq, ct, smooth_ct
        else:
            return seq, ct, None

    def label_smoothing(self, contact_map):
        # Define the kernel for the surrounding area
        kernel = np.array([[self.smooth, self.smooth, self.smooth],
                        [self.smooth, 0.0, self.smooth],
                        [self.smooth, self.smooth, self.smooth]])
        
        # Convolve the contact map with the kernel
        smoothed_map = convolve(contact_map.astype(float), kernel, mode='constant', cval=0.0)
        
        # Clip the values to be at most self.smooth
        smoothed_map = np.clip(smoothed_map, 0, self.smooth)
        
        # Where the original map is 1, we keep it 1
        smoothed_map[contact_map == 1] = 1.0
        
        return smoothed_map

class Augmentation:
    def __init__(self, select, replace, seed=42, mode='cov'):
        self.select = select
        self.replace = replace
        self.seed = seed
        self.mode = mode
        random.seed(self.seed)

    def __call__(self, seq, ct):
        if random.random() > self.select:
            return seq

        upper_triangle_indices = np.triu_indices_from(ct, k=1)

        all_pairs = np.column_stack(upper_triangle_indices)
        pairs = all_pairs[ct[upper_triangle_indices] == 1]

        seq_original = seq

        if self.mode == 'cov':
            for x, y in pairs:
                if random.random() < self.replace: 
                    if ((seq_original[x] == 'A') & (seq_original[y] == 'T'))|((seq_original[x] == 'T') & (seq_original[y] == 'A')):
                        if random.random() < 7.24/(7.24+46.3): # Wobble
                            if random.random() < 0.5:
                                seq = seq[:x] + 'T' + seq[x+1:]
                                seq = seq[:y] + 'G' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'G' + seq[x+1:]
                                seq = seq[:y] + 'T' + seq[y+1:]
                        else: # GC
                            if random.random() < 0.5:
                                seq = seq[:x] + 'G' + seq[x+1:]
                                seq = seq[:y] + 'C' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'C' + seq[x+1:]
                                seq = seq[:y] + 'G' + seq[y+1:]
                    elif ((seq_original[x] == 'C') & (seq_original[y] == 'G'))|((seq_original[x] == 'G') & (seq_original[y] == 'C')):
                        if random.random() < 7.24/(7.24+25.77): # Wobble
                            if random.random() < 0.5:
                                seq = seq[:x] + 'G' + seq[x+1:]
                                seq = seq[:y] + 'T' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'T' + seq[x+1:]
                                seq = seq[:y] + 'G' + seq[y+1:]
                        else: # AU
                            if random.random() < 0.5:
                                seq = seq[:x] + 'A' + seq[x+1:]
                                seq = seq[:y] + 'T' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'T' + seq[x+1:]
                                seq = seq[:y] + 'A' + seq[y+1:]
                    elif ((seq_original[x] == 'G') & (seq_original[y] == 'T'))|((seq_original[x] == 'T') & (seq_original[y] == 'G')):
                        if random.random() < 25.77/(25.77+46.3): # AU
                            if random.random() < 0.5:
                                seq = seq[:x] + 'A' + seq[x+1:]
                                seq = seq[:y] + 'T' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'T' + seq[x+1:]
                                seq = seq[:y] + 'A' + seq[y+1:]
                        else: # GC
                            if random.random() < 0.5:
                                seq = seq[:x] + 'G' + seq[x+1:]
                                seq = seq[:y] + 'C' + seq[y+1:]
                            else:
                                seq = seq[:x] + 'C' + seq[x+1:]
                                seq = seq[:y] + 'G' + seq[y+1:]
        
        elif self.mode == 'cg': 
            for x, y in pairs:
                if random.random() < self.replace:
                    if random.random() < 0.5:
                        seq = seq[:x] + 'C' + seq[x+1:]
                        seq = seq[:y] + 'G' + seq[y+1:]
                    else:
                        seq = seq[:x] + 'G' + seq[x+1:]
                        seq = seq[:y] + 'C' + seq[y+1:]

        return seq
