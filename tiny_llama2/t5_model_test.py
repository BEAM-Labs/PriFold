import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import open_dict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)
from tiny_llama2.t5_model import MyT5
from tiny_llama2.llama2_t5 import Llama2T5, ModelArgs

def unitest_t5():
    '''
    '''
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Origin T5 model
    print("Initializing a new model from scratch")
    config = AutoConfig.from_pretrained(
        f'{config_dir}/t5_config_l.json' # args.model.name,
    )
    model = MyT5(config)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # args = ModelArgs()
    # model = Llama2T5(args)
    # print(model)

    bs = 1
    seq_len = 128
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))

    target = torch.ones((bs, seq_len//2)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')

    global is_pre_cal_mask

    device = 'cpu'
    model.to(device)
    input = input.to(device)
    attn_mask = attn_mask.to(device)

    # cal time for 100 iterations
    import time
    start = time.time()
    iters = 1000
    for i in tqdm(range(iters)):
        output = model.forward(input_ids=input, attention_mask=attn_mask, labels=target)
    end = time.time()
    print(f'time: {(end - start) / iters}')


def cal_params():
    from tiny_llama2.dataset.mars.dataset_mars import tokenizer
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    print("Initializing a new model from scratch")

    config = AutoConfig.from_pretrained(
        f'{config_dir}/t5_config_l.json' # args.model.name,
    )
    config.vocab_size = tokenizer.vocab_size
    print(config.vocab_size)
    model = MyT5(config)
    print(f'number of encoder parameters: {sum(p.numel() for p in model.encoder.parameters())}')
    print(f'number of encoder parameters: {sum(p.numel() for p in model.parameters())}')

def cal_param_LT5():
    from tiny_llama2.llama2_t5 import get_t5_model_args
    args = get_t5_model_args(model_size='s')
    model = Llama2T5(args)
    print(model)
    print(f'number of encoder parameters: {sum(p.numel() for p in model.enc_layers.parameters())}')
    print(f'number of decoder parameters: {sum(p.numel() for p in model.dec_layers.parameters())}')
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')


if __name__ == '__main__':
    config_dir = '/public/home/taoshen/code/Projects/Zero2Hero/llama2/config/t5/'
    # cal_params()
    # unitest_t5()
    cal_param_LT5()
