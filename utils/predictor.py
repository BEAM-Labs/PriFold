from transformers.models.esm.modeling_esm import *
from collections import OrderedDict
import torch
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import random
from scipy.ndimage import convolve
    
class SSDataset_test(Dataset):
    def __init__(self, df, data_path, tokenizer):
        self.df = df
        self.data_path = data_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, str(file_name) + '.npy')
        ct = np.load(file_path)

        return seq, ct, file_name

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

class SSDataset_merge(Dataset):
    def __init__(self, df, data_path, tokenizer, aug=None, smooth=None):
        self.df = df
        self.data_paths = data_path  # 列表包含两个路径
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
        
        # 尝试从两个路径加载
        for path in self.data_paths:
            file_path = os.path.join(path, str(file_name) + '.npy')
            if os.path.exists(file_path):
                ct = np.load(file_path)
                break

        if self.aug:
            seq = self.aug(seq, ct)
        
        if self.smooth:
            smooth_ct = self.label_smoothing(ct)
            return seq, ct, smooth_ct
        else:
            return seq, ct, None

class Augmentation:
    def __init__(self, select, replace, seed=42, mode='cov'):
        self.select = select
        self.replace = replace
        self.seed = seed
        self.mode = mode
        random.seed(self.seed)

    def __call__(self, seq, ct):
        # 如果随机数大于 select，则不进行替换
        if random.random() > self.select:
            return seq

        # 获取矩阵的上三角部分的索引（不包括对角线）
        upper_triangle_indices = np.triu_indices_from(ct, k=1)

        # 使用上三角部分的索引筛选出元素为 1 的索引对
        all_pairs = np.column_stack(upper_triangle_indices)
        pairs = all_pairs[ct[upper_triangle_indices] == 1]

        seq_original = seq

        if self.mode == 'cov':
            for x, y in pairs:
                if random.random() < self.replace: # 由于online是用在replace(u,t)之后的，所以都是t
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
        
        elif self.mode == 'cg': # 配对部分全部替换为GC/CG
            for x, y in pairs:
                if random.random() < self.replace:
                    if random.random() < 0.5:
                        seq = seq[:x] + 'C' + seq[x+1:]
                        seq = seq[:y] + 'G' + seq[y+1:]
                    else:
                        seq = seq[:x] + 'G' + seq[x+1:]
                        seq = seq[:y] + 'C' + seq[y+1:]

        return seq


class SSDataset_test(Dataset):
    def __init__(self, df, data_path, tokenizer):
        self.df = df
        self.data_path = data_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['file_name']
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, str(file_name) + '.npy')
        ct = np.load(file_path)
        return seq, ct, name

class SSDataset_multi(Dataset):
    def __init__(self, df, data_path, data2_path ,tokenizer):
        self.df = df
        self.data_path = data_path
        self.data_path2 = data2_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, str(file_name) + '.npy')
        file_path2 = os.path.join(self.data_path2, str(file_name) + '.npy')
        ct = np.load(file_path)
        ct2 = np.load(file_path2)
        return seq, ct , ct2 

class SSDataset_mask(Dataset):
    def __init__(self, df, data_path, mask_path,tokenizer):
        self.df = df
        self.data_path = data_path
        self.mask_path = mask_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        file_name = row['file_name']
        file_path = os.path.join(self.data_path, str(file_name) + '.npy')
        ct = np.load(file_path)
        mask_path = os.path.join(self.mask_path, str(file_name) + '.npy')
        mask = np.load(mask_path)
        return seq, ct , mask
    
class SSCNNPredictor_dot_rnafm2(nn.Module):
    def __init__(self, extractor, esmconfig, label ,is_freeze = False):
        super(SSCNNPredictor_dot_rnafm2, self).__init__()
        self.extractor = extractor
        # self.esmconfig = esmconfig # esmconfig.hidden_size

        self.predictor = torch.nn.Sequential(torch.nn.Conv1d(esmconfig.hidden_size, 256, 1),torch.nn.Conv1d(256, label, 1))
        # self.predictor = nn.Linear(esmconfig.hidden_size , label)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # hidden_states = output.last_hidden_state
        hidden_states = output[1]
        x = self.predictor(hidden_states.transpose(1,2))

        return x

class PairwiseOnly(nn.Module):
    """
    contact predictor with pairwise concat
    """
    def __init__(self, backbone_args, embed_reduction=128):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__()
        self.embed_dim_in = backbone_args.hidden_size

        self.num_classes = 1
        self.symmetric = True

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None

        # self.proj = Lin2D(self.embed_dim_out * 2, self.num_classes)  #nn.Conv2d(self.embed_dim_out * 2, self.num_classes, kernel_size=1)

    def forward(self,  inputs):
        # embeddings = inputs.last_hidden_state
        embeddings = inputs
        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E


        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction


        # cosine similarity L*L pairwise concat
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3) # B, L, L, 2E
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2) # B, 2E, L, L
        
        return pairwise_concat_embedding

class PairwiseConcat(nn.Module):
    """
    contact predictor with pairwise concat
    """
    def __init__(self, backbone_args, embed_reduction=-1):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__()
        self.embed_dim_in = backbone_args.hidden_size

        self.num_classes = 1
        self.symmetric = True

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None

        self.proj = Lin2D(self.embed_dim_out * 2, self.num_classes)  #nn.Conv2d(self.embed_dim_out * 2, self.num_classes, kernel_size=1)

    def forward(self,  inputs):
        # embeddings = inputs.last_hidden_state
        embeddings = inputs
        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E


        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction


        # cosine similarity L*L pairwise concat
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3) # B, L, L, 2E
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2) # B, 2E, L, L

        output = self.proj(pairwise_concat_embedding) # B, 1, L, L

        # make output symmetric
        if self.symmetric == True:

            upper_triangular_output = torch.triu(output) 
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output
            output = torch.squeeze(output, dim=1) # B, L, L

        return output

class  PairwiseConcatWithResNet(PairwiseConcat):
    """
    contact predictor with pairwise concat + resnet
    reproduce from msa tranformer
    """
    def __init__(self, backbone_args, embed_reduction=128, num_res_layers=32):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__( backbone_args, embed_reduction,)


        self.embed_reduction = embed_reduction

        self.main_dim = 64

        first_layer = nn.Sequential(
            nn.Conv2d(self.embed_reduction*2, self.main_dim, kernel_size=1),
        )

        self.num_res_layers = num_res_layers
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)    # CJY 2022.12.20  2->1

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        # input B,2E,l,l  output B,1,l,l
        self.proj = nn.Sequential(layers)

class ResNet(nn.Module):
    def __init__(self, backbone_args, embed_reduction=128, pairwise=False):
        super(ResNet, self).__init__()
        self.embed_reduction = embed_reduction
        self.main_dim = 64

        if pairwise:
            first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction*2, self.main_dim, kernel_size=1),)
        else:
            first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction, self.main_dim, kernel_size=1),)

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        # input B,2E,l,l  output B,1,l,l
        self.proj = nn.Sequential(layers)
    
    def forward(self,x):
        x = self.proj(x)
        return x
    
from collections import OrderedDict
import torch
import torch.nn as nn

class MultiScaleResNet(nn.Module):
    def __init__(self, backbone_args, embed_reduction=128, pairwise=False):
        super(MultiScaleResNet, self).__init__()
        self.embed_reduction = embed_reduction
        self.main_dim = 64

        if pairwise:
            first_layer = nn.Sequential(nn.Conv2d(embed_reduction * 2, self.main_dim, kernel_size=1))
        else:
            first_layer = nn.Sequential(nn.Conv2d(embed_reduction, self.main_dim, kernel_size=1))

        self.num_res_layers = 32
        res_layers = []
        multifeature_layers = []

        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            block = MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation)
            res_layers.append(block)
            if i % 8 == 7:  # Every 8 layers, add a feature extraction for multiscale
                multifeature_layers.append(nn.AdaptiveAvgPool2d((i // 8 + 1, i // 8 + 1)))

        self.res_layers = nn.ModuleList(res_layers)
        self.multifeature_layers = nn.ModuleList(multifeature_layers)

        self.final_layer = nn.Conv2d(self.main_dim, self.main_dim, kernel_size=3, padding=1)

        # Calculate the size of concatenated features
        # feature_sizes = self.main_dim * sum([((i // 8 + 1) * (i // 8 + 1)) for i in range(self.num_res_layers) if i % 8 == 7])
        feature_sizes = sum([self.main_dim * ((i // 8 + 1) ** 2) for i in range(self.num_res_layers) if i % 8 == 7])
        self.fc = nn.Linear(feature_sizes, 1)  # Assuming the output is a single value

        self.first_layer = first_layer

    def forward(self, x):
        x = self.first_layer(x)
        features = []

        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i % 8 == 7:  # Corresponds to the layers where multiscale features are extracted
                idx = i // 8
                pooled_feature = self.multifeature_layers[idx](x)
                features.append(pooled_feature.view(pooled_feature.size(0), -1))

        # Concatenate all multi-scale features
        multiscale_features = torch.cat(features, dim=1)

        # Apply final transformation and fully connected layer
        x = self.final_layer(x)
        x = torch.flatten(x, 1)  # Flatten the features from final conv layer
        x = torch.cat((x, multiscale_features), dim=1)  # Concatenate with multi-scale features
        x = self.fc(x)

        return x

class ResNet_recycle(nn.Module):
    def __init__(self, backbone_args, num_cycles, embed_reduction=128, pairwise=False):
        super(ResNet_recycle, self).__init__()
        self.embed_reduction = embed_reduction
        self.main_dim = 64
        self.num_cycles = num_cycles
        
        if pairwise:
            self.first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction*2, self.main_dim, kernel_size=1),)
        else:
            self.first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction, self.main_dim, kernel_size=1),)

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        self.res_layers = nn.Sequential(*res_layers)

        self.final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)

        # layers = OrderedDict()
        # layers["first"] = first_layer
        # layers["resnet"] = res_layers
        # layers["final"] = final_layer

        # # input B,2E,l,l  output B,1,l,l
        # self.proj = nn.Sequential(layers)
    
    def forward(self,x, training):
        bs, dim, len, _ = x.shape
        x = self.first_layer(x)
        # breakpoint()
        if self.num_cycles>1:
            if training:
                n_cycles = torch.randint(2, self.num_cycles+1, [1]).item()
                # print(n_cycles)
            else:
                n_cycles = self.num_cycles
            
            latent = torch.zeros_like(x, requires_grad=True)
            with torch.no_grad():
                for _ in range(n_cycles-1):
                    res_latent = x.detach() + latent.detach()
                    latent = self.res_layers(res_latent)
            
            res_latent = x + latent
            x = self.res_layers(res_latent)
            x = self.final_layer(x)
        else:
            x = self.res_layers(x)
            x = self.final_layer(x)
        return x

class Axialattn(nn.Module):
    def __init__(self, backbone_args, embed_reduction=128):
        super(Axialattn, self).__init__()
        self.embed_dim_in = backbone_args.hidden_size

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None
        # Linear transformations for Q, K, and V
        # self.linear_q = torch.nn.Linear(input_dim, input_dim)
        # self.linear_k = torch.nn.Linear(input_dim, input_dim)
        # self.linear_v = torch.nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(x)
            hiddendim = self.embed_reduction
            
        x = embeddings.permute(0,2,1)
        # Linear transformations
        # q = self.linear_q(x)
        # k = self.linear_k(x)
        # v = self.linear_v(x)
        q = x.unsqueeze(3)
        k = x.unsqueeze(2)
        # Calculate attention weights
        # attention_weights = F.softmax(torch.bmm(q, k), dim=2)
        attn = torch.matmul(q,k) # b,e,l,l
        return attn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadSelfAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3) # B, H, L, E
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        # 计算注意力得分
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).float()) # B, H, L, L
        if mask is not None:
            # true_lengths = torch.sum(mask, dim=1)
            # expanded_mask = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[1]), dtype=tor
            # ch.float32)
            # for i in range(expanded_mask.shape[0]):
            #     expanded_mask[i, :true_lengths[i], :true_lengths[i]] = 1
            # mask = expanded_mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            # mask = mask.to(device=x.device)
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(energy, dim=-1)
        return attention

class AttnMapWithResnet(nn.Module):
    def __init__(self, backbone_args, embed_reduction=128):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super(AttnWithResnet, self).__init__()
        self.embed_dim_in = backbone_args.hidden_size
        self.self_attention = Axialattn()
        self.num_classes = 1
        self.symmetric = True

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None

        self.main_dim = 64

        first_layer = nn.Sequential(
            nn.Conv2d(self.embed_reduction, self.main_dim, kernel_size=1),
        )

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)    # CJY 2022.12.20  2->1

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        self.proj = nn.Sequential(layers)
    
    def forward(self, input):
        embeddings = input
        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E

        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction

        attention_output = self.self_attention(embeddings) # B, E, L, L

        output = self.proj(attention_output) # B, 1, L, L

        # make output symmetric
        if self.symmetric == True:

            upper_triangular_output = torch.triu(output) 
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output
            output = torch.squeeze(output, dim=1) # B, L, L

        return output

class AttnWithResnet(nn.Module):
    def __init__(self, backbone_args, embed_reduction=128):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super(AttnWithResnet, self).__init__()
        self.embed_dim_in = backbone_args.hidden_size
        self.num_classes = 1
        self.symmetric = True

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None
        
        self.main_dim = 64
        self.self_attention = MultiheadSelfAttention(embed_reduction, 8)
        first_layer = nn.Sequential(
            nn.Conv2d(8, self.main_dim, kernel_size=1),
        )

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)    # CJY 2022.12.20  2->1

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        self.proj = nn.Sequential(layers)
    
    def forward(self, input, mask=None):
        embeddings = input
        
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            # hiddendim = self.embed_reduction
        attention_output = self.self_attention(embeddings, mask) # B, E, L, L
        # B,h,l,l -> b,1,l,l
        output = self.proj(attention_output) # B, 1, L, L

        # make output symmetric
        if self.symmetric == True:

            upper_triangular_output = torch.triu(output) 
            lower_triangular_output = torch.triu(output, diagonal=1).permute(0, 1, 3, 2)
            output = upper_triangular_output + lower_triangular_output
            output = torch.squeeze(output, dim=1) # B, L, L

        return output


class MyBasicResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MyBasicResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # cjy commented
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)

        return out
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Lin2D(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.linear = nn.Linear(in_feat, 256) # in B,l,l,2d out B,l,l,256
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, out_feat)

    def forward(self, inputs):
        # inputs B, 2d, L, L
        inputs = inputs.permute(0, 2, 3, 1) # B, L, L, 2d

        outputs = self.linear2(self.relu(self.linear(inputs)))
        #outputs = self.linear(inputs)

        outputs = outputs.permute(0, 3, 1, 2) # B, out, L, L

        return outputs

class SSCNNPredictor_rnafm2(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False):
        super(SSCNNPredictor_rnafm2, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        # print(esmconfig.hidden_size)
        # self.cnn = renet_b16(myChannels=esmconfig.hidden_size, bbn=16)
        self.cnn = PairwiseConcatWithResNet(esmconfig)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # hidden_states = output.last_hidden_state
        hidden_states = output[1]
        ## L*ch-> LxL*ch
        hidden_states = hidden_states[:,1:-1,:]
        # matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)
        # matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d

        x = self.cnn(hidden_states)
        x = x.squeeze(-1)

        return x

class SSPredictor_PositionBias(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False, is_pairwise = True):
        super(SSPredictor_PositionBias, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        
        # expand or pairwiseconcat?
        if is_pairwise:
            self.expand = PairwiseOnly(esmconfig)
            # self.recycle_norm = nn.LayerNorm(256)
            self.cnn = ResNet(esmconfig, embed_reduction=256)
        else:
            self.expand = Axialattn(esmconfig)
            # self.recycle_norm = nn.LayerNorm(128)
            self.cnn = ResNet(esmconfig, embed_reduction=128)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        # seqs = data_dict['seqs']
        pos_bias = data_dict['pos_bias']
        pos_bias = pos_bias.detach()
        # pos_bias = get_posbias(seqs, input_ids.shape[1])
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        hidden_states = output[1]
        hidden_states = hidden_states[:,1:-1,:] # bs,l,e
        expanded = self.expand(hidden_states) # bs,e,l,l

        x=torch.clamp(expanded,min=0)
        
        # breakpoint()
        _, e, _, _ = expanded.shape
        pos_bias = pos_bias.unsqueeze(1).expand(-1, e, -1, -1)
        biased_x = torch.mul(x, pos_bias)
        
        x = self.cnn(biased_x)
        
        x = x.squeeze(-1)
        x = x.squeeze(1)
        return x

class SSPredictor_multif(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False, is_pairwise = False):
        super(SSPredictor_multif, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        
        # expand or pairwiseconcat?
        if is_pairwise:
            self.expand = PairwiseOnly(esmconfig)
            self.recycle_norm = nn.LayerNorm(256)
            self.cnn = MultiScaleResNet(esmconfig, embed_reduction=256)
        else:
            self.expand = Axialattn(esmconfig)
            self.recycle_norm = nn.LayerNorm(128)
            self.cnn = MultiScaleResNet(esmconfig, embed_reduction=128)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        # seqs = data_dict['seqs']
        # pos_bias = data_dict['pos_bias']
        # pos_bias = pos_bias.detach()
        # pos_bias = get_posbias(seqs, input_ids.shape[1])
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        hidden_states = output[1]
        hidden_states = hidden_states[:,1:-1,:] # bs,l,e
        expanded = self.expand(hidden_states) # bs,e,l,l

        x=torch.clamp(expanded,min=0)
        
        # breakpoint()
        # _, e, _, _ = expanded.shape
        # pos_bias = pos_bias.unsqueeze(1).expand(-1, e, -1, -1)
        # biased_x = torch.mul(x, pos_bias)
        
        x = self.cnn(x)
        
        x = x.squeeze(-1)
        x = x.squeeze(1)
        return x

class SSCNNPredictor_multiheadattn(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False):
        super(SSCNNPredictor_multiheadattn, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        self.cnn = AttnWithResnet(esmconfig)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # hidden_states = output.last_hidden_state
        hidden_states = output[1]
        ## L*ch-> LxL*ch
        hidden_states = hidden_states[:,1:-1,:]
        # matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)
        # matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d
        x = self.cnn(hidden_states, mask=attention_mask[:,1:-1])
        x = x.squeeze(-1)

        return x

def hook_fn(module, input, output):
    global hook_output
    hook_output = output
    return output

class SSCNNPredictor_attn(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False):
        super(SSCNNPredictor_attn, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        self.cnn = AttnWithResnet(esmconfig, embed_reduction=12)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        target_layer = self.extractor.layers[-1].attention.attn_dropout
        hook = target_layer.register_forward_hook(hook_fn)
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # out = torch.sum(hook_output, dim=1, keepdim=True)
        x = self.cnn(hook_output)
        x = x.squeeze(-1)

        return x    

class SSCNNPredictor_rnafm2_multi(nn.Module):
    def __init__(self, extractor, esmconfig, is_freeze = False):
        super(SSCNNPredictor_rnafm2_multi, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        # self.cnn = renet_b16(myChannels=esmconfig.hidden_size, bbn=16)
        self.cnn_canonical = PairwiseConcatWithResNet(esmconfig)
        self.cnn_noncanonical = PairwiseConcatWithResNet(esmconfig)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # hidden_states = output.last_hidden_state
        hidden_states = output[1]
        ## L*ch-> LxL*ch
        hidden_states = hidden_states[:,1:-1,:]
        # matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)
        # matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d

        x = self.cnn_canonical(hidden_states)
        x = x.squeeze(-1)
        x2 = self.cnn_noncanonical(hidden_states)
        x2 = x2.squeeze(-1)

        return x, x2

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, extractor, esmconfig, is_freeze = False, is_pairwise=False):
        super(TransformerModel, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        self.feature_size = feature_size
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)  # 解码到单个输出通道

        if is_pairwise:
            self.expand = PairwiseOnly(esmconfig)
            self.recycle_norm = nn.LayerNorm(256)
            self.cnn = MultiScaleResNet(esmconfig, embed_reduction=256)
        else:
            self.expand = Axialattn(esmconfig)
            self.recycle_norm = nn.LayerNorm(128)
            self.cnn = MultiScaleResNet(esmconfig, embed_reduction=128)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        # hidden_states = output.last_hidden_state
        hidden_states = output[1]
        ## L*ch-> LxL*ch
        hidden_states = hidden_states[:,1:-1,:]
        x = self.expand(hidden_states) # bs,e,l,l

        # 将输入数据的形状从 (b, l, l, e) 调整为 (l*l, b, e)
        b, e, l, _ = x.size()
        x = x.view(b, e, l*l).permute(2, 0, 1)
        
        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)
        
        # 解码到期望的输出尺寸 (l*l, b, 1) 然后调整形状回 (b, l, l)
        x = self.decoder(x)
        x = x.permute(1, 0, 2).view(b, l, l)
        
        return x

class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).chunk(3, dim=0)

        q = q.squeeze(0) * self.scale
        attn = (q @ k.squeeze(0).transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v.squeeze(0)).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class rnafm_cnn(nn.Module):
    def __init__(self, extractor, is_freeze = False):
        super(rnafm_cnn, self).__init__()
        self.extractor = extractor
        # print(esmconfig.hidden_size)
        # self.cnn = renet_b16(myChannels=esmconfig.hidden_size, bbn=16)
        self.expand = PairwiseOnly_fm(hidden_size=640, embed_reduction=128)
        self.cnn = ResNet_fm(embed_reduction=128)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        # input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        batch_tokens = data_dict['batch_tokens']
        # seqs = data_dict['seqs']
        # pos_bias = get_posbias(seqs, input_ids.shape[1])
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(batch_tokens, repr_layers=[12])
        else:
            output = self.extractor(batch_tokens, repr_layers=[12])

        token_embeddings = output["representations"][12] # [3, 71, 640] (bs, seq, dim)

        # hidden_states = output[1]
        # hidden_states = hidden_states[:,1:-1,:] # bs,l,e
        expanded = self.expand(token_embeddings) # bs,e,l,l

        x = self.cnn(expanded)
        
        x = x.squeeze(-1)
        x = x.squeeze(1)
        return x

class rnafm_cnn_bias(nn.Module):
    def __init__(self, extractor, is_freeze = False):
        super(rnafm_cnn_bias, self).__init__()
        self.extractor = extractor
        # print(esmconfig.hidden_size)
        # self.cnn = renet_b16(myChannels=esmconfig.hidden_size, bbn=16)
        self.expand = PairwiseOnly_fm(hidden_size=640, embed_reduction=128)
        self.cnn = ResNet_fm(embed_reduction=128)

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        # input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        batch_tokens = data_dict['batch_tokens']
        bias = data_dict['pos_bias']
        # seqs = data_dict['seqs']
        # pos_bias = get_posbias(seqs, input_ids.shape[1])
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(batch_tokens, repr_layers=[12])
        else:
            output = self.extractor(batch_tokens, repr_layers=[12])

        token_embeddings = output["representations"][12] # [3, 71, 640] (bs, seq, dim)

        # hidden_states = output[1]
        # hidden_states = hidden_states[:,1:-1,:] # bs,l,e
        expanded = self.expand(token_embeddings) # bs,e,l,l

        x = self.cnn(expanded)

        # breakpoint()
        x = x.squeeze(1)
        x *= bias
        
        # x = x.squeeze(-1)
        return x
    
class PairwiseOnly_fm(nn.Module):
    """
    contact predictor with pairwise concat
    """
    def __init__(self,hidden_size=256, embed_reduction=128):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super(PairwiseOnly_fm, self).__init__()
        self.embed_dim_in = hidden_size

        self.num_classes = 1
        self.symmetric = True

        self.embed_reduction = embed_reduction
        if self.embed_reduction != -1:
            self.embed_dim_out = self.embed_reduction
            self.pre_reduction = nn.Linear(self.embed_dim_in, self.embed_dim_out)
        else:
            self.embed_dim_out = self.embed_dim_in
            self.pre_reduction = None

        # self.proj = Lin2D(self.embed_dim_out * 2, self.num_classes)  #nn.Conv2d(self.embed_dim_out * 2, self.num_classes, kernel_size=1)

    def forward(self,  inputs):
        # embeddings = inputs.last_hidden_state
        embeddings = inputs
        if len(embeddings.size()) == 3:       # for seq
            batch_size, seqlen, hiddendim = embeddings.size()                    # B, L, E


        # embedding dim reduction
        if self.embed_reduction != -1:
            embeddings = self.pre_reduction(embeddings)
            hiddendim = self.embed_reduction


        # cosine similarity L*L pairwise concat
        embeddings = embeddings.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        embedding_T = embeddings.permute(0, 2, 1, 3)
        pairwise_concat_embedding = torch.cat([embeddings, embedding_T], dim=3) # B, L, L, 2E
        pairwise_concat_embedding = pairwise_concat_embedding.permute(0, 3, 1, 2) # B, 2E, L, L
        
        return pairwise_concat_embedding
    
class ResNet_fm(nn.Module):
    def __init__(self, embed_reduction=128, pairwise=True):
        super(ResNet_fm, self).__init__()
        self.embed_reduction = embed_reduction
        self.main_dim = 64

        if pairwise:
            first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction*2, self.main_dim, kernel_size=1),)
        else:
            first_layer = nn.Sequential(nn.Conv2d(self.embed_reduction, self.main_dim, kernel_size=1),)

        self.num_res_layers = 32
        res_layers = []
        for i in range(self.num_res_layers):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=self.main_dim, planes=self.main_dim, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        final_layer = nn.Conv2d(self.main_dim, 1, kernel_size=3, padding=1)

        layers = OrderedDict()
        layers["first"] = first_layer
        layers["resnet"] = res_layers
        layers["final"] = final_layer

        # input B,2E,l,l  output B,1,l,l
        self.proj = nn.Sequential(layers)
    
    def forward(self,x):
        x = self.proj(x)
        return x