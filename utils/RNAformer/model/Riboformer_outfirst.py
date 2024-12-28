import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from einops import rearrange, repeat

def is_package_installed(package_name):
    import pkg_resources
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    return package_name in installed_packages

# if is_package_installed('flash_attn'):
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat


class FlashAttention2d(nn.Module):
    def __init__(self, model_dim, num_head, softmax_scale,
                 zero_init, use_bias, initializer_range, n_layers):
        super().__init__()
        assert model_dim % num_head == 0
        assert model_dim % num_head == 0
        self.key_dim = model_dim // num_head
        self.value_dim = model_dim // num_head

        self.causal = False
        self.checkpointing = False

        if softmax_scale:
            self.softmax_scale = self.key_dim ** (-0.5)
        else:
            self.softmax_scale = None

        self.num_head = num_head

        self.Wqkv = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)

        self.out_proj = nn.Linear(model_dim, model_dim, bias=use_bias)

        self.initialize(zero_init, use_bias, initializer_range, n_layers)

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.Wqkv.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.out_proj.weight, 0.0)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, pair_act, attention_mask, bias):

        batch_size = pair_act.shape[0]
        seqlen = pair_act.shape[1]
        extended_batch_size = batch_size * seqlen

        qkv = self.Wqkv(pair_act)
        not_attention_mask = torch.logical_not(attention_mask)

        x_qkv = rearrange(qkv, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
        key_padding_mask = rearrange(not_attention_mask, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)

        x_unpad, indices, cu_seqlens, max_s = unpad_input(x_qkv, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=self.num_head)

        if self.training and self.checkpointing:
            output_unpad = torch.utils.checkpoint.checkpoint(flash_attn_varlen_qkvpacked_func,
                                                             x_unpad, cu_seqlens, max_s, 0.0, self.softmax_scale,
                                                             self.causal, False
                                                             )
        else:
            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad, cu_seqlens, max_s, 0.0,
                softmax_scale=self.softmax_scale, causal=self.causal
            )

        pre_pad_latent = rearrange(output_unpad, 'nnz h d -> nnz (h d)')
        padded_latent = pad_input(pre_pad_latent, indices, extended_batch_size, seqlen)
        output = rearrange(padded_latent, 'b f (h d) -> b f h d', h=self.num_head)

        output = rearrange(output, '(b s) f h d -> b s f (h d)', b=batch_size, f=seqlen, s=seqlen)

        return self.out_proj(output)


class Attention2d(nn.Module):
    def __init__(self, model_dim, num_head, softmax_scale,
                 precision, zero_init, use_bias,
                 initializer_range, n_layers, posbias=False):
        super().__init__()
        assert model_dim % num_head == 0
        assert model_dim % num_head == 0
        self.key_dim = model_dim // num_head
        self.value_dim = model_dim // num_head
        self.posbias = posbias

        if softmax_scale:
            self.softmax_scale = torch.sqrt(torch.FloatTensor([self.key_dim]))
        else:
            self.softmax_scale = False

        self.num_head = num_head
        self.model_dim = model_dim

        if precision == "fp32" or precision == 32 or precision == "bf16":
            self.mask_bias = -1e9
        elif precision == "fp16" or precision == 16:
            self.mask_bias = -1e4
        else:
            raise UserWarning(f"unknown precision: {precision} . Please us fp16, fp32 or bf16")

        self.Wqkv = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=use_bias)

        self.initialize(zero_init, use_bias, initializer_range, n_layers)

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.Wqkv.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.out_proj.weight, 0.0)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, pair_act, attention_mask, bias=None):

        batch_size = pair_act.size(0)
        N_seq = pair_act.size(1)
        N_res = pair_act.size(2)

        query, key, value = self.Wqkv(pair_act).split(self.model_dim, dim=3)

        query = query.view(batch_size, N_seq, N_res, self.num_head, self.key_dim).permute(0, 1, 3, 2, 4)
        # bs, N_seq, num_head, N_res, key_dim
        key = key.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(0, 1, 3, 4, 2)
        value = value.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(0, 1, 3, 2, 4)
        # breakpoint()
        attn_weights = torch.matmul(query, key) # bs, N_seq, num_head, N_res, N_res

        if self.softmax_scale:
            attn_weights *= self.softmax_scale.to(pair_act.device)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, None, :]
            attn_weights.masked_fill_(attention_mask, self.mask_bias)
        attn_weights = F.softmax(attn_weights, dim=-1)

        modified_attn_weights = attn_weights * bias

        weighted_avg = torch.matmul(modified_attn_weights, value).permute(0, 1, 3, 2, 4) # (bs, seq, head, res, value_dim) -> (bs, seq, res, head, value_dim)

        output = self.out_proj(weighted_avg.reshape(batch_size, N_seq, N_res, self.num_head * self.value_dim)) # bs, N_seq, N_res, head*value=256
        return output


class TriangleAttention(nn.Module):
    def __init__(self, model_dim, num_head, orientation, softmax_scale,
                 precision, zero_init, use_bias, flash_attn,
                 initializer_range, n_layers, posbias):
        super().__init__()

        self.model_dim = model_dim
        self.num_head = num_head

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.input_norm = nn.LayerNorm(model_dim, eps=1e-6)

        # breakpoint()
        if flash_attn :
            print('*********************flash_attn*********************')
            self.attn = FlashAttention2d(model_dim, num_head, softmax_scale, zero_init, use_bias, initializer_range, n_layers)
        else:
            self.attn = Attention2d(model_dim, num_head, softmax_scale,
                                    precision, zero_init, use_bias, initializer_range, n_layers, posbias)

    def forward(self, pair_act, pair_mask, bias, cycle_infer=False):

        assert len(pair_act.shape) == 4

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        pair_act = self.input_norm(pair_act)

        if self.training and not cycle_infer:
            pair_act = checkpoint(self.attn, pair_act, pair_mask, bias, use_reentrant=True)
        else:
            pair_act = self.attn(pair_act, pair_mask, bias)

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act


class AxialAttention(nn.Module):
    def __init__(self, config, orientation):
        super().__init__()

        self.model_dim = config['model_dim']
        self.num_head = config['num_head']
        assert config['model_dim'] % config['num_head'] == 0
        self.head_dim = config['model_dim'] // config['num_head']

        self.use_flash_attn = config['flash_attn']

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.input_norm = nn.LayerNorm(config['model_dim'], eps=config['ln_eps'], elementwise_affine=config['learn_ln'])

        if config['softmax_scale']:
            self.softmax_scale = self.head_dim ** (-0.5)
        else:
            self.softmax_scale = None

        precision = str(config['precision'])
        if "32" in config['precision'] or "bf16" in config['precision']:
            self.mask_bias = -1e9
        elif "16" in config['precision']:
            self.mask_bias = -1e4
        else:
            raise UserWarning(f"Unknown precision: {precision} . Please us fp16, fp32 or bf16")

        rotary_emb_dim = int(config['rotary_emb_fraction'] * self.head_dim)
        self.rotary_emb = RotaryEmbedding(rotary_emb_dim, )

        self.Wqkv = nn.Linear(config['model_dim'], 3 * config['model_dim'], bias=config['use_bias'])
        self.out_proj = nn.Linear(config['model_dim'], config['model_dim'], bias=config['use_bias'])

        self.initialize(config['zero_init'], config['use_bias'], config['initializer_range'], config['n_layers'])

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.Wqkv.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.Wqkv.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.out_proj.weight, 0.0)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    @staticmethod
    def _multihead_attn(query, key, value, attention_mask, mask_bias, softmax_scale, batch_size, seqlen, num_head,
                        head_dim):

        query = query.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 2, 4)
        key = key.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 4, 2)
        value = value.view(batch_size, seqlen, seqlen, num_head, head_dim).permute(0, 1, 3, 2, 4)

        attn_weights = torch.matmul(query, key)

        if softmax_scale:
            attn_weights *= softmax_scale

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, None, :]
            attn_weights.masked_fill_(attention_mask, mask_bias)
        attn_weights = F.softmax(attn_weights, dim=-1)

        weighted_avg = torch.matmul(attn_weights, value).permute(0, 1, 3, 2, 4)
        output = weighted_avg.reshape(batch_size, seqlen, seqlen, num_head * head_dim)

        return output

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        assert len(pair_act.shape) == 4

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        batch_size = pair_act.shape[0]
        seqlen = pair_act.shape[1]
        extended_batch_size = batch_size * seqlen

        pair_act = self.input_norm(pair_act)

        query, key, value = self.Wqkv(pair_act).split(self.model_dim, dim=3)

        freqs_h = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device), cache_key=seqlen)
        freqs_w = self.rotary_emb(torch.linspace(-1, 1, steps=seqlen).to(pair_act.device), cache_key=seqlen)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        query = apply_rotary_emb(freqs, query)
        key = apply_rotary_emb(freqs, key)

        if self.use_flash_attn :
            qkv = torch.cat((query, key, value), dim=-1)
            not_attention_mask = torch.logical_not(pair_mask)

            x_qkv = rearrange(qkv, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen, s=seqlen)
            key_padding_mask = rearrange(not_attention_mask, 'b s f ... -> (b s) f ...', b=batch_size, f=seqlen,
                                         s=seqlen)

            x_unpad, indices, cu_seqlens, max_s = unpad_input(x_qkv, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=self.num_head)

            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad, cu_seqlens, max_s, 0.0,
                softmax_scale=self.softmax_scale, causal=False
            )

            pre_pad_latent = rearrange(output_unpad, 'nnz h d -> nnz (h d)')
            padded_latent = pad_input(pre_pad_latent, indices, extended_batch_size, seqlen)
            output = rearrange(padded_latent, '(b s) f hd -> b s f hd', b=batch_size, f=seqlen, s=seqlen)

        else:
            output = self._multihead_attn(query, key, value, pair_mask, self.mask_bias, self.softmax_scale,
                                          batch_size, seqlen, self.num_head, self.head_dim)

        pair_act = self.out_proj(output)

        if self.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act


class FeedForward(nn.Module):

    def __init__(self, config):
        super(FeedForward, self).__init__()

        ff_dim = int(config['ff_factor'] * config['model_dim'])

        self.glu = config['use_glu']

        if self.glu:
            ff_dim_2 = np.exp2(np.ceil(np.log2(256 * 4 / 3))).astype(int)
            ff_dim_1 = ff_dim_2 * 2
        else:
            ff_dim_1, ff_dim_2 = ff_dim, ff_dim

        self.input_norm = nn.LayerNorm(config['model_dim'], eps=config['ln_eps'], elementwise_affine=config['learn_ln'])

        self.linear_1 = nn.Linear(config['model_dim'], ff_dim_1, bias=config['use_bias'])
        self.linear_2 = nn.Linear(ff_dim_2, config['model_dim'], bias=config['use_bias'])
        self.act = nn.SiLU()

        self.initialize(config['zero_init'], config['use_bias'], config['initializer_range'], config['n_layers'])

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.linear_1.bias, 0.0)
            nn.init.constant_(self.linear_2.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.linear_2.weight, 0.0)
        else:
            nn.init.normal_(self.linear_2.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, x):

        x = self.input_norm(x)

        if self.glu:
            x = self.linear_1(x)
            x, gate = x.chunk(2, dim=-1)
            x = self.act(gate) * x
        else:
            x = self.act(self.linear_1(x))

        return self.linear_2(x)


class ConvFeedForward(nn.Module):

    def __init__(self, config):
        super(ConvFeedForward, self).__init__()

        ff_dim = int(config['ff_factor'] * config['model_dim'])

        self.zero_init = config['zero_init']

        self.input_norm = nn.GroupNorm(1, config['model_dim'], affine=config['learn_ln'])

        if config['ff_kernel'] == 1:
            self.conv1 = nn.Conv2d(config['model_dim'], ff_dim, kernel_size=1, bias=config['use_bias'])
            self.conv2 = nn.Conv2d(ff_dim, config['model_dim'], kernel_size=1, bias=config['use_bias'])
        else:
            self.conv1 = nn.Conv2d(config['model_dim'], ff_dim, bias=config['use_bias'], kernel_size=config['ff_kernel'],
                                   padding=(config['ff_kernel'] - 1) // 2)
            self.conv2 = nn.Conv2d(ff_dim, config['model_dim'], bias=config['use_bias'], kernel_size=config['ff_kernel'],
                                   padding=(config['ff_kernel'] - 1) // 2)

        self.act = nn.SiLU()

        self.initialize(config['zero_init'], config['use_bias'], config['initializer_range'], config['n_layers'])

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.conv1.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.conv1.bias, 0.0)
            nn.init.constant_(self.conv2.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.conv2.weight, 0.0)
        else:
            nn.init.normal_(self.conv2.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        x = self.input_norm(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        return x

class RNAformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attn_pair_row = TriangleAttention(config['model_dim'], config['num_head'], 'per_row', config['softmax_scale'],
                                                config['precision'], config['zero_init'], config['use_bias'],
                                                config['flash_attn'],
                                                config['initializer_range'], config['n_layers'], config['posbias'])
        self.attn_pair_col = TriangleAttention(config['model_dim'], config['num_head'], 'per_column',
                                                config['softmax_scale'],
                                                config['precision'], config['zero_init'], config['use_bias'],
                                                config['flash_attn'],
                                                config['initializer_range'], config['n_layers'], config['posbias'])

        self.pair_dropout_row = nn.Dropout(p=config['resi_dropout'] / 2)
        self.pair_dropout_col = nn.Dropout(p=config['resi_dropout'] / 2)

        if config['ff_kernel']:
            self.pair_transition = ConvFeedForward(config)
        else:
            self.pair_transition = FeedForward(config)

        self.res_dropout = nn.Dropout(p=config['resi_dropout'])

    def forward(self, pair_act, pair_mask, bias, cycle_infer=False):

        pair_act = pair_act + self.pair_dropout_row(self.attn_pair_row(pair_act, pair_mask, bias, cycle_infer))
        pair_act = pair_act + self.pair_dropout_col(self.attn_pair_col(pair_act, pair_mask, bias, cycle_infer))
        pair_act = pair_act + self.res_dropout(self.pair_transition(pair_act))

        return pair_act


class RNAformerStack(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.output_ln = nn.LayerNorm(config['model_dim'], eps=config['ln_eps'], elementwise_affine=config['learn_ln'])

        module_list = []
        for idx in range(config['n_layers']):
            layer = RNAformerBlock(config=config)
            module_list.append(layer)
        self.layers = nn.ModuleList(module_list)

    def forward(self, pair_act, pair_mask, bias, cycle_infer=False):
        # 加 (b,1,1,l,l)
        bias = bias.unsqueeze(1).unsqueeze(2)
        # 乘 (b,l,l,1),广播
        # bias = bias.unsqueeze(-1)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                pair_act = layer(pair_act, pair_mask, bias, cycle_infer=cycle_infer)
            else:
                pair_act = layer(pair_act, pair_mask, torch.zeros_like(bias), cycle_infer=cycle_infer)

        pair_act = self.output_ln(pair_act)

        return pair_act



class PosEmbedding(nn.Module):
    def __init__(self, vocab, model_dim, max_len, rel_pos_enc, initializer_range):

        super().__init__()

        self.rel_pos_enc = rel_pos_enc
        self.max_len = max_len

        self.embed_seq = nn.Embedding(vocab, model_dim)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([model_dim // 2])), requires_grad=False)

        if rel_pos_enc:
            self.embed_pair_pos = nn.Linear(max_len, model_dim, bias=False)
        else:
            self.embed_pair_pos = nn.Linear(model_dim, model_dim, bias=False)

            pe = torch.zeros(max_len, model_dim)
            position = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2).type(torch.FloatTensor) * -(math.log(10000.0) / model_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            pe = torch.nn.Parameter(pe, requires_grad=False)
            self.register_buffer('pe', pe)

        self.initialize(initializer_range)  #

    def initialize(self, initializer_range):

        nn.init.normal_(self.embed_seq.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.embed_pair_pos.weight, mean=0.0, std=initializer_range)

    def relative_position_encoding(self, src_seq):

        residue_index = torch.arange(src_seq.size()[1], device=src_seq.device).expand(src_seq.size())
        rel_pos = F.one_hot(torch.clip(residue_index, min=0, max=self.max_len - 1), self.max_len)

        if isinstance(self.embed_pair_pos.weight, torch.cuda.BFloat16Tensor):
            rel_pos = rel_pos.type(torch.bfloat16)
        elif isinstance(self.embed_pair_pos.weight, torch.cuda.HalfTensor):
            rel_pos = rel_pos.half()
        else:
            rel_pos = rel_pos.type(torch.float32)

        pos_encoding = self.embed_pair_pos(rel_pos)
        return pos_encoding

    def forward(self, src_seq):

        seq_embed = self.embed_seq(src_seq) * self.scale

        if self.rel_pos_enc:
            seq_embed = seq_embed + self.relative_position_encoding(src_seq)
        else:
            seq_embed = seq_embed + self.embed_pair_pos(self.pe[:, :src_seq.size(1)])

        return seq_embed

class PairwiseOnly(nn.Module):
    """
    contact predictor with pairwise concat
    """
    def __init__(self, embed_reduction=128):
        """
        :param depth_reduction: mean, first, (adaptive)
        """
        super().__init__()
        self.embed_dim_in = 1056

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

class EmbedSequence2Matrix(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pos_embedding = config['pos_embedding']

        if config['pos_embedding']:
            self.src_embed_1 = PosEmbedding(config['seq_vocab_size'], config['model_dim'], config['max_len'],
                                            config['rel_pos_enc'], config['initializer_range'])
            self.src_embed_2 = PosEmbedding(config['seq_vocab_size'], config['model_dim'], config['max_len'],
                                            config['rel_pos_enc'], config['initializer_range'])
        else:
            self.src_embed_1 = nn.Embedding(config['seq_vocab_size'], config['model_dim'])
            self.src_embed_2 = nn.Embedding(config['seq_vocab_size'], config['model_dim'])
            # Note: we need to ensure that `model_dim` is correctly accessed and potentially cast to float before sqrt
            self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([config['model_dim'] // 2])), requires_grad=False)

        self.norm = nn.LayerNorm(config['model_dim'], eps=config['ln_eps'], elementwise_affine=config['learn_ln'])
    def forward(self, src_seq):
        seq_1_embed = self.src_embed_1(src_seq)
        seq_2_embed = self.src_embed_2(src_seq)

        if not self.pos_embedding:
            seq_1_embed = seq_1_embed * self.scale
            seq_2_embed = seq_2_embed * self.scale

        pair_latent = seq_1_embed.unsqueeze(1) + seq_2_embed.unsqueeze(2)

        pair_latent = self.norm(pair_latent)

        return pair_latent

class RiboFormer(nn.Module):

    def __init__(self, config, extractor, is_freeze):
        super().__init__()
        print(config)
        self.extractor = extractor
        # config = config['RNAformer']
        self.model_dim = config['model_dim']
        self.is_freeze = is_freeze

        if config.get("cycling", False):  # Default to False if 'cycling' key is not present
            self.initialize_cycling(config['cycling'])  # Dictionary-style access
            print('cycling')
        else:
            self.cycling = False

        # self.seq2mat_embed = EmbedSequence2Matrix(config)
        self.RNAformer = RNAformerStack(config)
        self.expand = PairwiseOnly()

        # if not config.get("pdb_flag", None):
        #     self.pdb_embedding = nn.Linear(1, config['model_dim'], bias=True)
        #     self.use_pdb = True
        # else:
        #     self.use_pdb = False

        # Using the `get` method with a default value of `None` if 'binary_output' is not found
        if not config.get("binary_output", None):
            self.output_mat = nn.Linear(config['model_dim'], 1, bias=True)
        else:
            self.output_mat = nn.Linear(config['model_dim'], 2, bias=False)

        self.initialize(initializer_range=config['initializer_range'])

    def initialize(self, initializer_range):

        nn.init.normal_(self.output_mat.weight, mean=0.0, std=initializer_range)

    def initialize_cycling(self, cycle_steps):
        import random
        self.cycling = True
        self.cycle_steps = cycle_steps
        self.recycle_pair_norm = nn.LayerNorm(self.model_dim, elementwise_affine=True)
        self.trng = torch.Generator()
        self.trng.manual_seed(random.randint(1, 10000))

    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)

        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]

        assert isinstance(pair_mask, torch.BoolTensor) or isinstance(pair_mask, torch.cuda.BoolTensor)
        return torch.bitwise_not(pair_mask)

    @torch.no_grad()
    def cycle_riboformer(self, pair_act, pair_mask, bias):
        latent = self.RNAformer(pair_act=pair_act, pair_mask=pair_mask, bias=bias, cycle_infer=True)
        return latent.detach()

    # def forward(self, src_seq, src_len, pdb_sample, max_cycle=0):
    def forward(self, data_dict, max_cycle=0):
        # pair_mask = self.make_pair_mask(src_seq, src_len)
        input_ids = data_dict['input_ids']
        attention_mask = data_dict['attention_mask']
        bias = data_dict['pos_bias']

        # pair_latent = self.seq2mat_embed(src_seq)
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)

        hidden_states = output[1]
        pair_latent = self.expand(hidden_states) # b, e, l ,l
        pair_latent = pair_latent.permute(0,2,3,1) # b, l, l, e

        pair_mask = self.make_pair_mask(input_ids, data_dict['seq_len'])

        latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, bias=bias, cycle_infer=False)

        logits = self.output_mat(latent)

        return logits