import random
import math
import inspect
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from prifold.utils.attn_utils import get_extended_attention_mask
from prifold.llama2 import (ModelArgs, Attention, precompute_freqs_cis, TransformerBlock, RMSNorm,
                          apply_rotary_emb, repeat_kv, apply_rotary_emb_qkv)
from torch import nn
from torch.nn import CrossEntropyLoss
from prifold.t5_model import Seq2SeqLMOutput, EncoderOutput
from copy import deepcopy

def get_t5_model_args(model_size, vocab_size = 21, dropout = 0.15, n_dec_layers = 1):

    model_args = \
        get_model_args(model_size, False, vocab_size=vocab_size, dropout=dropout)

    model_args = ModelArgs(**model_args)

    model_args.n_dec_layers = n_dec_layers

    return model_args

class Llama2T5(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        params.is_decoder = False
        self.enc_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.enc_layers.append(TransformerBlock(layer_id, params))

        decdoer_params = deepcopy(params)
        decdoer_params.is_decoder = True
        self.dec_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_dec_layers):
            self.dec_layers.append(TransformerBlock(layer_id, decdoer_params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call.
        # This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_layers(self,
                       input_ids,
                       attention_mask,
                       layers,
                       is_decoder,
                       encoder_h,):

        ckpt_inter = 1

        _bsz, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)
        if attention_mask is not None:
            h = (h * attention_mask.unsqueeze(-1)).to(h.dtype)

        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # extend attention mask to get the score
        if attention_mask is None:
            attention_mask = torch.ones((_bsz, seqlen), device=h.device, dtype=h.dtype)

        extended_attn_mask = get_extended_attention_mask(attention_mask,(_bsz, seqlen), h.device, h.dtype, is_decoder=is_decoder)

        for c, layer in enumerate(layers):

            if (c+1) % ckpt_inter == 0:
                h,_ = checkpoint(layer, h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)
            else:
                h,_ = layer(h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)

        return h

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> (torch.Tensor, torch.Tensor):
        '''
        :param tokens: (bsz, seqlen) LongTensor of input token indices
        :param attention_mask: (bsz, seqlen) padding mask for attention
        '''

        encoder_output = self.forward_layers(input_ids, attention_mask, self.enc_layers, is_decoder = False, encoder_h=None)

        h = self.forward_layers(decoder_input_ids, decoder_attention_mask, self.dec_layers, is_decoder = True, encoder_h=encoder_output)
        # LM head
        h = self.norm(h)

        logits = self.output(h)

        self.last_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            self.last_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=self.last_loss,
            logits=logits,
            encoder_outputs=EncoderOutput(hidden_states=encoder_output, attention_mask=attention_mask),
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.

class Llama2T5_encoder(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        params.is_decoder = False
        self.enc_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.enc_layers.append(TransformerBlock(layer_id, params))

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward_layers(self,
                       input_ids,
                       attention_mask,
                       layers,
                       is_decoder,
                       encoder_h = None ):

        ckpt_inter = 1

        _bsz, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)
        if attention_mask is not None:
            h = (h * attention_mask.unsqueeze(-1)).to(h.dtype)

        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # extend attention mask to get the score
        if attention_mask is None:
            attention_mask = torch.ones((_bsz, seqlen), device=h.device, dtype=h.dtype)

        extended_attn_mask = get_extended_attention_mask(attention_mask, (_bsz, seqlen), h.device, h.dtype,
                                                         is_decoder=is_decoder)

        for c, layer in enumerate(layers):
            h_pre = h

            if (c + 1) % ckpt_inter == 0:
                h, _ = checkpoint(layer, h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)
            else:
                h, _ = layer(h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)

            if self.training and random.random() < self.params.layer_dropout:
                # dirty imp for layer dropout
                h = h_pre + 0.0 * h

        return h

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attn_mask: Optional[torch.FloatTensor] = None,
                ) -> (torch.Tensor, torch.Tensor):
        '''
        :param tokens: (bsz, seqlen) LongTensor of input token indices
        :param attention_mask: (bsz, seqlen) padding mask for attention
        '''

        encoder_output = self.forward_layers(input_ids, attn_mask, self.enc_layers, is_decoder=False)

        return None, encoder_output

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.


def load_encoder_model(ckpt_path, model_size, vocab_size, device = 'cpu', dropout = None, layer_dropout = None, from_scratch = False):
    # init from a model saved in a specific directory

    print(f"loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = get_t5_model_args(model_size=model_size, vocab_size=vocab_size)

    if dropout is not None:
        gptconf.dropout = dropout
    if layer_dropout is not None:
        gptconf.layer_dropout = layer_dropout

    model = Llama2T5_encoder(gptconf)

    if from_scratch:
        print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
        return model

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
    return model



def unitest_CrossAttention():
    args = ModelArgs()
    args.dim = 512

    freqs_cos, freqs_sin = precompute_freqs_cis(dim = args.dim // args.n_heads, end = 512)

    model = Attention(args)
    seqlen = 32
    seqlen2 = 48

    freqs_cos, freqs_sin = freqs_cos[:seqlen], freqs_sin[:seqlen]
    x = torch.randn(1, seqlen, args.dim)
    attn_mask = torch.ones(1, seqlen, 1)
    kv_state = torch.randn(1, seqlen2, args.dim)
    out = model.forward(x, attn_mask, freqs_cos, freqs_sin, kv_state = kv_state,)
    print(out[0].shape)

    args.is_decoder = True
    decoder_layers = TransformerBlock(layer_id = 0, args = args)
    print(decoder_layers)

    out = decoder_layers.forward(x, attn_mask, freqs_cos, freqs_sin, kv_state = kv_state)
    print(out[0].shape)
