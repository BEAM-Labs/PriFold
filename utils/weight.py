import math
import pandas as pd
import torch
import os
from tqdm import tqdm
import numpy as np

class Weight():
    def __init__(self, K, dataset, level, data_dir):
        self.K = K
        self.dataset = dataset
        self.level = level
        self.data_dir = data_dir
        self.pos, self.neg = self.collect(level, dataset)
        print(self.pos, self.neg)
        
    def get_weight(self, mode, pos, neg):
        if mode == 'pos':
            return math.log10(neg/pos + self.K)
        elif mode == 'neg':
            return math.log10(pos/neg + self.K)
    
    def collect(self, num_categories, dataset):
        if dataset[:4] == 'spot':
            spotdir = f'{self.data_dir}/PDB_spotrna/'
            df = pd.read_csv(spotdir+'pdbRNA.csv')
            ctPath = spotdir + 'ct/'
            name = df[df['data_name']== 'TR1']['file_name']
        elif dataset[:7] == 'PDB_125':
            ctPath = f'{self.data_dir}/PDB_125/ct/'
            df = pd.read_csv(f'{self.data_dir}/PDB_125/PDB_125.csv')
            name = df[df['data_name']== 'TR1']['file_name']
        elif dataset[:5] == 'bprna':
            return [179734, 143520, 74188, 55670, 41038, 32576, 31484, 34300, 42124, 29610], [50253247, 45018644, 39988220, 34476308, 29039960, 24180772, 18667296, 13265854, 7984046, 2643190]
            df = pd.read_csv(f'{self.data_dir}/bprna/bpRNA.csv')
            ctPath = f'{self.data_dir}/bprna/ct/TR0/'
            name = df[df['data_name']== 'TR0']['file_name']
        elif dataset[:9] == 'archiveii':
            return [138992, 49954, 19476, 27036, 17558, 12036, 10550, 17514, 24590, 25696], [33234358, 29904578, 26462198, 22869260, 19293798, 15945944, 12361340, 8790052, 5285542, 1742070]
            df = pd.read_csv(f'{self.data_dir}/archiveii.csv')
            ctPath = f'{self.data_dir}/archiveii/'
            name = df[df['data_name']== 'TR1']['file_name']
        elif dataset[:11] == 'rnastralign':
            return [447678, 344790, 141392, 130246, 61800, 12528, 9756, 22982, 121910, 118672], [104258840, 93272330, 82726958, 71463142, 60309310, 50197246, 38694418, 27513584, 16502426, 5429684]
            ctPath = '/public_aim/yangchenchen/RNAStralign/ct/'
            df = pd.read_csv(f'{self.data_dir}/rnastralign.csv')
            df = df[df['seq'].str.len() < 490]
            name= df[df['data_name']== 'tr']['file_name']

        name = name.tolist()
        # print(name)
        
        pos = [0] * num_categories
        neg = [0] * num_categories

        for i in tqdm(range(len(name))):
            if os.path.exists(ctPath + name[i] + '.npy'):
                ct = np.load(ctPath + name[i] + '.npy')
                L = ct.shape[0]
                
                for category in range(num_categories):
                    lower_bound = category * L / num_categories
                    upper_bound = (category + 1) * L / num_categories
                    for j in range(L):
                        for k in range(L):
                            if lower_bound <= abs(j-k) < upper_bound:
                                if ct[j][k] == 1:
                                    pos[category] += 1
                                else:
                                    neg[category] += 1
        return pos, neg

    def weight_split(self, labels):
        L = labels.shape[0]
        weight = np.ones((L, L))

        diff = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        labels = labels.detach().cpu().numpy()
        
        # if self.level == 10:
        for i in range(self.level):
            weight[(i*L/self.level <= diff) & (diff < (i+1)*L/self.level)] = np.where(labels[(i*L/self.level <= diff) & (diff < (i+1)*L/self.level)], self.get_weight('pos', self.pos[i], self.neg[i]), self.get_weight('neg', self.pos[i], self.neg[i]))
        weight = torch.tensor(weight)
        return weight
    
        if self.dataset == 'spotrna':
            # PDB_ALL log level5
            weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 3822166, 6301960726), self.get_weight('neg', 3822166, 6301960726))
            weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 245206, 4905902580), self.get_weight('neg', 245206, 4905902580))
            weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 53782, 3504269054), self.get_weight('neg', 53782, 3504269054))
            weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 35960, 2102702850), self.get_weight('neg', 35960, 2102702850))
            weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 75602, 700932814), self.get_weight('neg', 75602, 700932814))
        
        # elif self.dataset == 'spotrna':
        #     # spotrna log level5
        #     weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 3760, 376193), self.get_weight('neg', 3760, 376193))
        #     weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 1636, 294868), self.get_weight('neg', 1636, 294868))
        #     weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 1066, 211218), self.get_weight('neg', 1066, 211218))
        #     weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 750, 126512), self.get_weight('neg', 750, 126512))
        #     weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 1110, 41594), self.get_weight('neg', 1110, 41594))
        
        elif self.dataset == 'archiveii':
            weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 188946, 63138936), self.get_weight('neg', 188946, 63138936))
            weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 46512, 49331458), self.get_weight('neg', 46512, 49331458))
            weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 29594, 35239742), self.get_weight('neg', 29594, 35239742))
            weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 28064, 21151392), self.get_weight('neg', 28064, 21151392))
            weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 50286, 7027612), self.get_weight('neg', 50286, 7027612))
        
        
        elif self.dataset == 'bprna' or self.dataset == 'bprna_sample':
            # bprna log level5
            weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 323254, 95271891), self.get_weight('neg', 323254, 95271891))
            weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 129858, 74464528), self.get_weight('neg', 129858, 74464528))
            weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 73614, 53220732), self.get_weight('neg', 73614, 53220732))
            weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 65784, 31933150), self.get_weight('neg', 65784, 31933150))
            weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 71734, 10627236), self.get_weight('neg', 71734, 10627236))
        
        # else:
        #     # PDB_ALL log level5
        #     weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 3822166, 6301960726), self.get_weight('neg', 3822166, 6301960726))
        #     weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 245206, 4905902580), self.get_weight('neg', 245206, 4905902580))
        #     weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 53782, 3504269054), self.get_weight('neg', 53782, 3504269054))
        #     weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 35960, 2102702850), self.get_weight('neg', 35960, 2102702850))
        #     weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 75602, 700932814), self.get_weight('neg', 75602, 700932814))

        elif self.dataset == 'dssr':
            # dssr log level5
            weight[diff < L/5] = np.where(labels[diff < L/5], self.get_weight('pos', 9520, 1901977), self.get_weight('neg', 9520, 1901977))
            weight[(L/5 <= diff) & (diff < 2*L/5)] = np.where(labels[(L/5 <= diff) & (diff < 2*L/5)], self.get_weight('pos', 4122, 1490880), self.get_weight('neg', 4122, 1490880))
            weight[(2*L/5 <= diff) & (diff < 3*L/5)] = np.where(labels[(2*L/5 <= diff) & (diff < 3*L/5)], self.get_weight('pos', 2090, 1063728), self.get_weight('neg', 2090, 1063728))
            weight[(3*L/5 <= diff) & (diff < 4*L/5)] = np.where(labels[(3*L/5 <= diff) & (diff < 4*L/5)], self.get_weight('pos', 1706, 639724), self.get_weight('neg', 1706, 639724))
            weight[(4*L/5 <= diff) & (diff < 5*L/5)] = np.where(labels[(4*L/5 <= diff) & (diff < 5*L/5)], self.get_weight('pos', 2684, 211408), self.get_weight('neg', 2684, 211408))
        
        # dssr log 10
        # weight[diff < L/10] = np.where(labels[diff < L/10], get_weight('pos', 4578, 1003455), get_weight('neg', 4578, 1003455))
        # weight[(L/10 <= diff) & (diff < 2*L/10)] = np.where(labels[(L/10 <= diff) & (diff < 2*L/10)], get_weight('pos', 4942, 898522), get_weight('neg', 4942, 898522))
        # weight[(2*L/10 <= diff) & (diff < 3*L/10)] = np.where(labels[(2*L/10 <= diff) & (diff < 3*L/10)], get_weight('pos', 2424, 800056), get_weight('neg', 2424, 800056))
        # weight[(3*L/10 <= diff) & (diff < 4*L/10)] = np.where(labels[(3*L/10 <= diff) & (diff < 4*L/10)], get_weight('pos', 1698, 690824), get_weight('neg', 1698, 690824))
        # weight[(4*L/10 <= diff) & (diff < 5*L/10)] = np.where(labels[(4*L/10 <= diff) & (diff < 5*L/10)], get_weight('pos', 1134, 579640), get_weight('neg', 1134, 579640))
        # weight[(5*L/10 <= diff) & (diff < 6*L/10)] = np.where(labels[(5*L/10 <= diff) & (diff < 6*L/10)], get_weight('pos', 956, 484088), get_weight('neg', 956, 484088))
        # weight[(6*L/10 <= diff) & (diff < 7*L/10)] = np.where(labels[(6*L/10 <= diff) & (diff < 7*L/10)], get_weight('pos', 822, 374892), get_weight('neg', 822, 374892))
        # weight[(7*L/10 <= diff) & (diff < 8*L/10)] = np.where(labels[(7*L/10 <= diff) & (diff < 8*L/10)], get_weight('pos', 884, 264832), get_weight('neg', 884, 264832))
        # weight[(8*L/10 <= diff) & (diff < 9*L/10)] = np.where(labels[(8*L/10 <= diff) & (diff < 9*L/10)], get_weight('pos', 1340, 159430), get_weight('neg', 1340, 159430))
        # weight[(9*L/10 <= diff) & (diff < 10*L/10)] = np.where(labels[(9*L/10 <= diff) & (diff < 10*L/10)], get_weight('pos', 1344, 51978), get_weight('neg', 1344, 51978))
        
        weight = torch.tensor(weight)
        return weight