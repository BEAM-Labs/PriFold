import torch
import numpy as np
from utils.predictor import SSDataset, SSDataset_test
import pandas as pd

def load_data(args, tokenizer, aug=None, smooth=None):
    if args.mode == 'bprna':
        ## bprna data ##
        df = pd.read_csv(f'{args.data_dir}/bprna/bpRNA.csv')

        df = df[df['seq'].str.len() < 490]
        df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
        df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/bprna/ct/TR0', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/bprna/ct/VL0', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=None, smooth=None)
    
    if args.mode == 'bprna-test':
        df = pd.read_csv(f'{args.data_dir}/bprna/bpRNA.csv')

        df = df[df['seq'].str.len() < 490]
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)

        train_dataset = None
        val_dataset = None
        test_dataset = SSDataset_test(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer)
    
    elif args.mode == 'archiveii':
        df = pd.read_csv(f'{args.data_dir}/archiveII/archiveII.csv')
        train_dataset = None
        val_dataset = None
        test_dataset = SSDataset(df, data_path=f'{args.data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)
    
    elif args.mode == 'bprna_sample':
        ## bprna data ##
        df = pd.read_csv(f'{args.data_dir}/bprna/bpRNA_sample.csv')

        df = df[df['seq'].str.len() < 490]
        
        df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)

        # df = df.sample(frac=0.2, random_state=42)
        df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/bprna/ct/TR0', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/bprna/ct/VL0', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=None, smooth=None)

    elif args.mode == 'rnastralign':

        df_str = pd.read_csv(f'{args.data_dir}/RNAStrAlign/rnastralign.csv')
        df_arc = pd.read_csv(f'{args.data_dir}/archiveII/archiveII.csv')

        df_str = df_str[df_str['seq'].str.len() < 490]
        df_arc = df_arc[df_arc['seq'].str.len() < 490]

        df_train = df_str[df_str['data_name'] == 'tr'].reset_index(drop=True)
        df_val = df_str[df_str['data_name'] == 'ts'].reset_index(drop=True)
        df_test = df_arc.reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)

    return train_dataset, val_dataset, test_dataset

def get_posbias(seqs, max_len, scale):
    # 初始化矩阵，所有值默认为0
    posbias = np.ones((len(seqs), max_len-2, max_len-2), dtype=float)
    pair_scores = {
        ('A', 'T'): 3,
        ('T', 'A'): 3,
        ('G', 'C'): 6,
        ('C', 'G'): 6,
        ('G', 'T'): 1,
        ('T', 'G'): 1
    }
    for i, seq in enumerate(seqs):
        # 遍历序列中的每个可能的配对位置
        for j in range(len(seq)):
            for k in range(len(seq)):
                # 取序列中的两个核苷酸
                nucleotide1 = seq[j]
                nucleotide2 = seq[k]
                # 检查这对核苷酸是否在配对规则中
                if (nucleotide1, nucleotide2) in pair_scores:
                    # 在矩阵的相应位置上设置分数
                    posbias[i, j, k] = pair_scores[(nucleotide1, nucleotide2)] * scale + 1.0
    posbias = np.pad(posbias, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=1)
    return torch.Tensor(posbias)