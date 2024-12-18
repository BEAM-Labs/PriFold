import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from utils.predictor import SSDataset, SSDataset_Smoothing
import pandas as pd
from utils.aug import aug, aug_pair

def split_len(logits, labels, splitLen=100):
    near_logits, far_logits, near_labels, far_labels = [], [], [], []
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if abs(i-j) < splitLen:
                near_logits.append(logits[i, j])
                near_labels.append(labels[i, j])
            else:
                far_logits.append(logits[i, j])
                far_labels.append(labels[i, j])
    near_logits = torch.stack(near_logits)
    near_labels = torch.stack(near_labels)
    near_loss = nn.BCEWithLogitsLoss()(near_logits, near_labels)

    if len(far_logits) > 0:
        far_logits = torch.stack(far_logits)
        far_labels = torch.stack(far_labels)
        far_loss = nn.BCEWithLogitsLoss()(far_logits, far_labels)
    
    return near_loss, far_loss

def equal5(logits, labels):
    L = labels.shape[0]

    diff = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    labels = labels.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    f1Split5 = [[] for _ in range(5)]
    logSplit5 = [[] for _ in range(5)]
    labSplit5 = [[] for _ in range(5)]
    test = []

    for i in range(5):
        lower = i*L/5
        upper = (i+1)*L/5
        for j in range(L):
            for k in range(L):
                if lower <= abs(j-k) < upper:
                    logSplit5[i].append(logits[j, k])
                    labSplit5[i].append(labels[j, k])

    for i in range(5):
        logSplit5[i] = np.stack(logSplit5[i])
        labSplit5[i] = np.stack(labSplit5[i])
        
    for i in range(5):
        if labSplit5[i].sum() != 0:
            logSplit5[i] = torch.FloatTensor(logSplit5[i])
            prob = torch.sigmoid(logSplit5[i])
            pred = (prob > 0.5).float()
            f1Split5[i].append(f1_score(labSplit5[i].reshape(-1), pred.reshape(-1)))
            
    return f1Split5[0], f1Split5[1], f1Split5[2], f1Split5[3], f1Split5[4]

def split_closeness(logits, labels):
    log10, lab10, log5, lab5, log2, lab2, log1, lab1 = [], [], [], [], [], [], [], []
    L = logits.shape[0]
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if abs(i-j) < L/10:
                log10.append(logits[i, j])
                lab10.append(labels[i, j])
            elif abs(i-j) < L/5:
                log5.append(logits[i, j])
                lab5.append(labels[i, j])
            elif abs(i-j) < L/2:
                log2.append(logits[i, j])
                lab2.append(labels[i, j])
            else:
                log1.append(logits[i, j])
                lab1.append(labels[i, j])
    if len(log10) > 0:
        log10 = torch.stack(log10)
        lab10 = torch.stack(lab10)
        prob = torch.sigmoid(log10)
        pred = (prob > 0.5).float()
        loss10 = nn.BCEWithLogitsLoss()(log10, lab10)
        f110 = f1_score(lab10.cpu().reshape(-1), pred.cpu().reshape(-1))
        pre10 = precision_score(lab10.cpu().reshape(-1), pred.cpu().reshape(-1))
        rec10 = recall_score(lab10.cpu().reshape(-1), pred.cpu().reshape(-1))
    else:
        loss10 = 'None'
        f110 = 'None'
        pre10 = 'None'
        rec10 = 'None'
    if len(log5) > 0:
        log5 = torch.stack(log5)
        lab5 = torch.stack(lab5)
        prob = torch.sigmoid(log5)
        pred = (prob > 0.5).float()
        loss5 = nn.BCEWithLogitsLoss()(log5, lab5)
        f15 = f1_score(lab5.cpu().reshape(-1), pred.cpu().reshape(-1))
        pre5 = precision_score(lab5.cpu().reshape(-1), pred.cpu().reshape(-1))
        rec5 = recall_score(lab5.cpu().reshape(-1), pred.cpu().reshape(-1))
    else:
        loss5 = 'None'
        f15 = 'None'
        pre5 = 'None'
        rec5 = 'None'
    if len(log2) > 0:
        log2 = torch.stack(log2)
        lab2 = torch.stack(lab2)
        prob = torch.sigmoid(log2)
        pred = (prob > 0.5).float()
        loss2 = nn.BCEWithLogitsLoss()(log2, lab2)
        f12 = f1_score(lab2.cpu().reshape(-1), pred.cpu().reshape(-1))
        pre2 = precision_score(lab2.cpu().reshape(-1), pred.cpu().reshape(-1))
        rec2 = recall_score(lab2.cpu().reshape(-1), pred.cpu().reshape(-1))
    else:
        loss2 = 'None'
        f12 = 'None'
        pre2 = 'None'
        rec2 = 'None'
    if len(log1) > 0:
        log1 = torch.stack(log1)
        lab1 = torch.stack(lab1)
        prob = torch.sigmoid(log1)
        pred = (prob > 0.5).float()
        loss1 = nn.BCEWithLogitsLoss()(log1, lab1)
        f11 = f1_score(lab1.cpu().reshape(-1), pred.cpu().reshape(-1))
        pre1 = precision_score(lab1.cpu().reshape(-1), pred.cpu().reshape(-1))
        rec1 = recall_score(lab1.cpu().reshape(-1), pred.cpu().reshape(-1))
    else:
        loss1 = 'None'
        f11 = 'None'
        pre1 = 'None'
        rec1 = 'None'

    return loss10, loss5, loss2, loss1, f110, f15, f12, f11, pre10, pre5, pre2, pre1, rec10, rec5, rec2, rec1

def load_data(args, tokenizer, aug=None, smooth=None):
    # if smooth is not None:
    #     SSDataset = SSDataset_Smoothing
    # else:
    #     SSDataset = SSDataset
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

    # elif args.mode == 'pdb':
    #     df = pd.read_csv(f'{args.pdb_dir}/pdbRNA.csv')

    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode == 'bprna_can':
    #     df = pd.read_csv(f'{args.bprna_dir}/new_bpRNA.csv')
        
    #     df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.bprna_dir}/can_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.bprna_dir}/can_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.bprna_dir}/can_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    elif args.mode == 'spotrna':
        ## bprna data ##
        df = pd.read_csv(f'{args.data_dir}/PDB_spotrna/pdbRNA.csv')

        df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
        df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer,aug=None, smooth=None)

    # elif args.mode == 'javapdb':
    #     ## bprna data ##
    #     df = pd.read_csv(f'{args.javapdb_dir}/java_nc_fix_remove_clu.csv')

    #     df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.javapdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.javapdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.javapdb_dir}/ct', tokenizer=tokenizer, aug=aug, smooth=smooth) 

    # elif args.mode == 'javapdb_wonc':
    #     ## bprna data ##
    #     df = pd.read_csv(f'{args.javapdb_dir}/java_fix_remove_clu.csv')

    #     df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.javapdb_dir}/ct2', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.javapdb_dir}/ct2', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.javapdb_dir}/ct2', tokenizer=tokenizer, aug=aug, smooth=smooth) 

    # elif args.mode == 'dssr':
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_updated.csv')
    #     df = df[df['seq'].str.len() < 512]
    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode == 'dssr_nc':
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_unargs.model_nc_updated.csv')

    #     df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr_unargs.model_nc/ct_unargs.model', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr_unargs.model_nc/ct_unargs.model', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr_unargs.model_nc/ct_unargs.model', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode[:-1] == 'spot_fold':
    #     fold = args.mode[-1]
    #     print('---------------  Start spot fold '+fold+'  -------------')
    #     raw = pd.read_csv('/mnt/data/aim/yangchenchen/projects/datasets/PDB_dataset/pdb_raw.csv')

    #     fold = int(fold)
    #     raw_train = raw[~raw.index.isin(range(fold*43,(fold+1)*43))]
        
    #     df_pdb_train = raw_train.reset_index(drop=True)
    #     df_pdb_val = pd.DataFrame()
    #     df_pdb_test = raw.iloc[fold*43:(fold+1)*43].reset_index(drop=True)
        
    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/aim/yangchenchen/projects/datasets/PDB_dataset/PDB_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/aim/yangchenchen/projects/datasets/PDB_dataset/PDB_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/aim/yangchenchen/projects/datasets/PDB_dataset/PDB_ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
        
    # elif args.mode[:-1] == 'bprna_aug_unpair':
    #     select = args.mode[-1]
    #     print('---------------  Start bprna aug select '+select+'  -------------')
    #     df = pd.read_csv(f'{args.data_dir}/aug/bpRNA_aug/bpRNA_aug_unpair'+select+'.csv')
        
    #     df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        
    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/bprna/ct/TR0', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/bprna/ct/VL0', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode[:-1] == 'bprna_aug_pair':
    #     select = args.mode[-1]
    #     print('---------------  Start bprna aug select '+select+'  -------------')
    #     df = pd.read_csv(f'{args.data_dir}/aug/bpRNA_aug/bpRNA_aug_pair'+select+'.csv')
        
    #     df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
    #     df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
    #     df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        
    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/bprna/ct/TR0', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/bprna/ct/VL0', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=aug, smooth=smooth)
        
    # elif args.mode[:9] == 'dssr_fold':
    #     fold = args.mode[9]
    #     if args.mode[10:15] == '_pair':
    #         replace = args.mode[15]
    #         if args.mode[16:-1] == '_select':
    #             select = args.mode[-1]
    #             print('---------------  Start replace '+replace+' fold'+fold+' select '+select+'  -------------')
    #             raw = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw.csv')
    #             raw_aug = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw_aug_pair'+replace+'.csv')

    #             # test: [fold*121:(fold+1)*121]
    #             fold = int(fold)
    #             raw_train = raw[~raw.index.isin(range(fold*121,(fold+1)*121))]
    #             raw_aug_train = raw_aug[~raw_aug.index.isin(range(fold*121,(fold+1)*121))]
    #             select = '0.'+select
    #             select = float(select)
    #             raw_aug_train = raw_aug_train.sample(frac=select, random_state=42)
                
    #             df_pdb_train = pd.concat([raw_train, raw_aug_train]).reset_index(drop=True)
    #             df_pdb_val = pd.DataFrame()
    #             df_pdb_test = raw.iloc[fold*121:(fold+1)*121].reset_index(drop=True)
        
    #     elif args.mode[10:-1] == '_half':
    #         replace = args.mode[-1]
    #         print('---------------  Start half replace '+replace+' fold'+fold+'  -------------')
    #         raw = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw.csv')
    #         raw_aug = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw_aug_pair'+replace+'.csv')
            
    #         # test: [fold*121:(fold+1)*121]
    #         fold = int(fold)
    #         raw_train = raw[~raw.index.isin(range(fold*121,(fold+1)*121))]
    #         raw_aug_train = raw_aug[~raw_aug.index.isin(range(fold*121,(fold+1)*121))]
    #         raw_aug_train = raw_aug_train.sample(frac=0.5, random_state=42)
            
    #         df_pdb_train = pd.concat([raw_train, raw_aug_train]).reset_index(drop=True)
    #         df_pdb_val = pd.DataFrame()
    #         df_pdb_test = raw.iloc[fold*121:(fold+1)*121].reset_index(drop=True)
        
    #     elif args.mode[10:-1] == '_unpair':
    #         replace = args.mode[-1]
    #         print('---------------  Start replace '+replace+' fold'+fold+'  -------------')
    #         raw = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw.csv')
    #         raw_aug = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw_aug_unpair'+replace+'.csv')
            
    #         # test: [fold*121:(fold+1)*121]
    #         fold = int(fold)
    #         raw_train = raw[~raw.index.isin(range(fold*121,(fold+1)*121))]
    #         raw_aug_train = raw_aug[~raw_aug.index.isin(range(fold*121,(fold+1)*121))]
            
    #         df_pdb_train = pd.concat([raw_train, raw_aug_train]).reset_index(drop=True)
    #         df_pdb_val = pd.DataFrame()
    #         df_pdb_test = raw.iloc[fold*121:(fold+1)*121].reset_index(drop=True)
        
    #     else: # baseline 5fold
    #         raw = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_fold/dssr_raw.csv')
    #         print('---------------  Start baseline fold'+fold+'  -------------')
            
    #         # test: [fold*121:(fold+1)*121]
    #         fold = int(fold)
    #         raw_train = raw[~raw.index.isin(range(fold*121,(fold+1)*121))]
            
    #         df_pdb_train = raw_train.reset_index(drop=True)
    #         df_pdb_val = pd.DataFrame()
    #         df_pdb_test = raw.iloc[fold*121:(fold+1)*121].reset_index(drop=True)
        
        
    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode == 'dssr_aug_both_all':
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_aug_both_all.csv')
        
    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode == 'dssr_aug_both':
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/dssr_aug_both.csv')
        
    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode[:-1] == 'dssr_aug':
    #     print('---------------  Start aug '+ args.mode[-1] + '  -------------')
    #     aug(args.mode[-1])
        
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/aug/dssr_aug/dssr_aug' + args.mode[-1] + '.csv')

    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode[:-1] == 'dssr_aug_pair':
    #     print('---------------  Start aug '+ args.mode[-1] + '  -------------')
    #     aug_pair(args.mode[-1])
        
    #     df = pd.read_csv('/mnt/data/aim/yangchenchen/projects/aug/dssr_aug/dssr_aug_pair' + args.mode[-1] + '.csv')

    #     df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_pdb_train, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_pdb_val, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_pdb_test, data_path='/mnt/data/oss_beijing/yangchenchen/dssr/dssr/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
    # elif args.mode[:-1] == 'spot_aug_unpair':
    #     print('---------------  Start aug '+ args.mode[-1] + '  -------------')
    #     # aug(args.mode[-1])
        
    #     df = pd.read_csv(f'{args.data_dir}/aug/spot_aug/spot_aug' + args.mode[-1] + '.csv')

    #     df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)  
    
    # elif args.mode[:13] == 'spot_aug_pair':
    #     print('---------------  Start aug '+ args.mode[-3:] + '  -------------')
    #     # aug_pair(args.mode[-1])
        
    #     df = pd.read_csv(f'{args.data_dir}/aug/spot_aug/spot_aug_pair' + args.mode[-3:] + '.csv')

    #     df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/PDB_spotrna/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)  

    elif args.mode[:16] == 'PDB_125':
        
        df = pd.read_csv(f'{args.data_dir}/PDB_125/PDB_125.csv')

        df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
        df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
        df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=None, smooth=None)

    # elif args.mode[:16] == 'PDB_125_aug_pair':
    #     print('---------------  Start aug '+ args.mode[-3:] + '  -------------')
    #     # aug_pair(args.mode[-1])
        
    #     df = pd.read_csv(f'{args.data_dir}/aug/PDB_125_aug/PDB_125_aug_pair' + args.mode[-3:] + '.csv')

    #     df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/PDB_125/ct', tokenizer=tokenizer, aug=aug, smooth=smooth)  

    elif args.mode == 'archiveii':
        df = pd.read_csv(f'{args.data_dir}/archiveii.csv')

        df = df[df['seq'].str.len() < 490]
        # df = df[df['seq'].str.len() < 512]
        df_train = df[df['data_name'] == 'TR1'].reset_index(drop=True)
        df_val = df[df['data_name'] == 'VL1'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS1'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=None, smooth=None)
    
    # elif args.mode[:-1] == 'archiveii_aug_unpair':
    #     print('---------------  Start aug '+ args.mode[-1] + '  -------------')
    #     # aug(args.mode[-1])
        
    #     df = pd.read_csv(f'{args.data_dir}/aug/archiveii_aug/archiveii_aug' + args.mode[-1] + '.csv')

    #     df = df[df['seq'].str.len() < 512]
    #     df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)

    # elif args.mode[:-1] == 'archiveii_aug_pair':
    #     print('---------------  Start aug '+ args.mode[-1] + '  -------------')
    #     # aug_pair(args.mode[-1])
        
    #     df = pd.read_csv(f'{args.data_dir}/aug/archiveii_aug/archiveii_aug_pair' + args.mode[-1] + '.csv')

    #     df = df[df['seq'].str.len() < 490]
    #     df_train = df[df['data_name']=='TR1'].reset_index(drop=True)
    #     df_val = df[df['data_name']=='VL1'].reset_index(drop=True)
    #     df_test = df[df['data_name']=='TS1'].reset_index(drop=True)

    #     train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
    #     test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveii', tokenizer=tokenizer, aug=aug, smooth=smooth)
    
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

    elif args.mode == 'rnastralign_s':
        df_str = pd.read_csv(f'{args.data_dir}/RNAStrAlign/rnastralign.csv')
        df_arc = pd.read_csv(f'{args.data_dir}/archiveII/archiveII.csv')

        df_str = df_str[df_str['seq'].str.len() < 490]
        df_arc = df_arc[df_arc['seq'].str.len() < 490]

        df_train = df_str[df_str['data_name'] == 'tr'].reset_index(drop=True)
        df_train = df_train.sample(frac=0.1, random_state=42)
        df_val = df_str[df_str['data_name'] == 'ts'].reset_index(drop=True)
        df_test = df_arc.reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)

    return train_dataset, val_dataset, test_dataset