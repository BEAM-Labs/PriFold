import pandas as pd
import random
import numpy as np
import os
from tqdm import tqdm
random.seed(0)
# REPLACE = 0.1
def aug(REPLACE):
    REPLACE = '0.'+REPLACE
    REPLACE = float(REPLACE)
    
    df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_updated.csv')
    # df = pd.read_csv('/home/yangchenchen/projects/datasets/bpRNA/bpRNA_.csv')

    ctPath = '/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct/'
    # ctPath = '/mnt/data/aim/yangchenchen/projects/datasets/bpRNA/ct/'
    dstFile = '/mnt/data/aim/yangchenchen/projects/dssr_aug'+str(REPLACE)[-1]+'.csv'
    
    if REPLACE == 0.0:
        REPLACE = 1

    # 选择序列
    selected_indices = random.sample(range(len(df)), k=int(len(df)))
    selected_df = df.loc[selected_indices].copy()

    for index, row in selected_df.iterrows():
        name = row['file_name']
        seq = row['seq']
        if row['data_name'] == 'TR1':
            if os.path.exists(ctPath + name + '.npy'):
                ct = np.load(ctPath + name + '.npy')
                for i in range(ct.shape[0]):
                    if ct[i].sum() == 0:
                        if random.random() < REPLACE: # replace概率替换
                            if seq[i] == 'A':
                                seq = seq[:i] + random.choice(['U', 'C', 'G']) + seq[i+1:]
                            elif seq[i] == 'U':
                                seq = seq[:i] + random.choice(['A', 'C', 'G']) + seq[i+1:]
                            elif seq[i] == 'C':
                                seq = seq[:i] + random.choice(['A', 'U', 'G']) + seq[i+1:]
                            elif seq[i] == 'G':
                                seq = seq[:i] + random.choice(['A', 'U', 'C']) + seq[i+1:]
                if row['seq'] != seq:
                    selected_df.at[index, 'seq'] = seq
                else:
                    selected_df.drop(index, inplace=True)
            else:
                selected_df.drop(index, inplace=True)
        else:
            selected_df.drop(index, inplace=True)

    all_df = pd.concat([df, selected_df], ignore_index=True)
    all_df.to_csv(dstFile)


def aug_pair(REPLACE):
    REPLACE = '0.'+REPLACE
    REPLACE = float(REPLACE)
    
    df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_updated.csv')
    # df = pd.read_csv('/home/yangchenchen/projects/datasets/bpRNA/bpRNA_.csv')

    ctPath = '/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct/'
    # ctPath = '/mnt/data/aim/yangchenchen/projects/datasets/bpRNA/ct/'
    dstFile = '/mnt/data/aim/yangchenchen/projects/dssr_aug_pair'+str(REPLACE)[-1]+'.csv'
    
    if REPLACE == 0.0:
        REPLACE = 1

    # 选择序列
    selected_indices = random.sample(range(len(df)), k=int(len(df)))
    selected_df = df.loc[selected_indices].copy()

    for index, row in selected_df.iterrows():
        name = row['file_name']
        seq = row['seq']
        if row['data_name'] == 'TR1':
            if os.path.exists(ctPath + name + '.npy'):
                ct = np.load(ctPath + name + '.npy')
                pairs = np.argwhere(ct == 1)
                for x, y in pairs:
                    if random.random() < REPLACE:
                        if ((seq[x] == 'A') & (seq[y] == 'U'))|((seq[x] == 'U') & (seq[y] == 'A')):
                            if random.random() < 7.24/(7.24+46.3): # Wobble
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                            else: # GC
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'C' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'C' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                        elif ((seq[x] == 'C') & (seq[y] == 'G'))|((seq[x] == 'G') & (seq[y] == 'C')):
                            if random.random() < 7.24/(7.24+25.77): # Wobble
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                            else: # AU
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'A' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'A' + seq[y+1:]
                        elif ((seq[x] == 'G') & (seq[y] == 'U'))|((seq[x] == 'U') & (seq[y] == 'G')):
                            if random.random() < 25.77/(25.77+46.3): # AU
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'A' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'A' + seq[y+1:]
                            else: # GC
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'C' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'C' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                if row['seq'] != seq:
                    selected_df.at[index, 'seq'] = seq
                else:
                    selected_df.drop(index, inplace=True)
            else:
                selected_df.drop(index, inplace=True)
        else:
            selected_df.drop(index, inplace=True)

    all_df = pd.concat([df, selected_df], ignore_index=True)
    all_df.to_csv(dstFile)

def aug_both(rPair, rUnpair):
    df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_updated.csv')
    
    ctPath = '/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct/'
    
    dstPath = '/mnt/data/aim/yangchenchen/projects/dssr_aug_both_all.csv'
    
    selected_indices = random.sample(range(len(df)), k=int(len(df)))
    selected_df = df.loc[selected_indices].copy()

    for index, row in selected_df.iterrows():
        name = row['file_name']
        seq = row['seq']
        if row['data_name'] == 'TR1':
            if os.path.exists(ctPath + name + '.npy'):
                ct = np.load(ctPath + name + '.npy')
                pairs = np.argwhere(ct == 1)
                for x, y in pairs:
                    if random.random() < rPair:
                        if ((seq[x] == 'A') & (seq[y] == 'U'))|((seq[x] == 'U') & (seq[y] == 'A')):
                            if random.random() < 7.24/(7.24+46.3): # Wobble
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                            else: # GC
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'C' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'C' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                        elif ((seq[x] == 'C') & (seq[y] == 'G'))|((seq[x] == 'G') & (seq[y] == 'C')):
                            if random.random() < 7.24/(7.24+25.77): # Wobble
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                            else: # AU
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'A' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'A' + seq[y+1:]
                        elif ((seq[x] == 'G') & (seq[y] == 'U'))|((seq[x] == 'U') & (seq[y] == 'G')):
                            if random.random() < 25.77/(25.77+46.3): # AU
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'A' + seq[x+1:]
                                    seq = seq[:y] + 'U' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'U' + seq[x+1:]
                                    seq = seq[:y] + 'A' + seq[y+1:]
                            else: # GC
                                if random.random() < 0.5:
                                    seq = seq[:x] + 'G' + seq[x+1:]
                                    seq = seq[:y] + 'C' + seq[y+1:]
                                else:
                                    seq = seq[:x] + 'C' + seq[x+1:]
                                    seq = seq[:y] + 'G' + seq[y+1:]
                for i in range(ct.shape[0]):
                    if ct[i].sum() == 0:
                        if random.random() < rUnpair: # replace概率替换
                            if seq[i] == 'A':
                                seq = seq[:i] + random.choice(['U', 'C', 'G']) + seq[i+1:]
                            elif seq[i] == 'U':
                                seq = seq[:i] + random.choice(['A', 'C', 'G']) + seq[i+1:]
                            elif seq[i] == 'C':
                                seq = seq[:i] + random.choice(['A', 'U', 'G']) + seq[i+1:]
                            elif seq[i] == 'G':
                                seq = seq[:i] + random.choice(['A', 'U', 'C']) + seq[i+1:]
                if row['seq'] != seq:
                    selected_df.at[index, 'seq'] = seq
                else:
                    selected_df.drop(index, inplace=True)
            else:
                selected_df.drop(index, inplace=True)
        else:
            selected_df.drop(index, inplace=True)
    
    pair = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_aug_pair2.csv')
    unpair = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_aug3.csv')
    
    all_df = pd.concat([df, selected_df, pair, unpair], ignore_index=True).drop_duplicates()
    all_df.to_csv(dstPath)
