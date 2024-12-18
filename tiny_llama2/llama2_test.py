from tiny_llama2.llama2 import *
from tqdm import tqdm
import numpy as np

def unitest_flash_attn():
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    '''
    print("Loading a pretrained model")
    ckpt_path = ('/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/'
                 'mars_run-encoder-mars-s-add_attn_mask-2023_08_21_22_18_02/ckpt.pt')
    model = load_model(ckpt_path, device='cpu')
    '''

    print("Initializing a new model from scratch")
    model_args = get_model_args('l', 'encoder', vocab_size=3000)
    gptconf = ModelArgs(**model_args)
    gptconf.layer_dropout = 0.5
    model = Transformer(gptconf)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    bs = 16
    seq_len = 512
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))

    target = torch.ones((bs, seq_len)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')

    is_cuda = False
    if is_cuda:
        model.half()
        model.to('cuda:0')
        input = input.to('cuda:0')
        attn_mask = attn_mask.to('cuda:0')
        target = target.to('cuda:0')

    # cal time for 100 iterations
    import time
    start = time.time()
    iters = 100
    for i in tqdm(range(iters)):
        output = model.forward(input, attn_mask, target)
        model.last_loss.backward()

    end = time.time()
    print(f'time: {(end - start) / iters}')

    #12192MiB time: 0.32055283784866334
    #12092MiB

def unitest_transfomer():

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    '''
    print("Loading a pretrained model")
    ckpt_path = ('/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/'
                 'mars_run-encoder-mars-s-add_attn_mask-2023_08_21_22_18_02/ckpt.pt')
    model = load_model(ckpt_path, device='cpu')
    '''

    print("Initializing a new model from scratch")
    model_args = get_model_args('lxxx', 'encoder', vocab_size=3000)
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    bs = 16
    seq_len = 1024
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))

    target = torch.ones((bs, seq_len)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')

    global is_pre_cal_mask

    # is_pre_cal_mask = True
    # model.is_decoder = True
    # output = model.forward(input, attn_mask, target)

    device = 'cpu'

    model.to(device)
    input = input.to(device)
    attn_mask = attn_mask.to(device)

    # cal time for 100 iterations
    import time
    start = time.time()
    iters = 1000
    for i in tqdm(range(iters)):
        output = model.forward(input, attn_mask)
    end = time.time()
    print(f'time: {(end - start) / iters}')

    # Flash attention in pytorch 2.0
    # 3760MiB  time: 0.01803 0.0178 0.01828
    # 3908MiB  time: 0.01885 0.0187 0.01874

    # 3036MiB  time: 0.07310
    # 3212MiB

    # is_pre_cal_mask = False
    # model.is_decoder = True
    # output = model.forward(input, attn_mask, target)
    # print(output)
    #
    # is_pre_cal_mask = True
    # model.is_decoder = False
    # output = model.forward(input, attn_mask, target)
    # print(output)
    #
    # is_pre_cal_mask = False
    # model.is_decoder = False
    # output = model.forward(input, attn_mask, target)
    # print(output)

def cal_params():
    from tiny_llama2.dataset.mars.dataset_mars import tokenizer

    for model_size in ['s']: #,'m','l','lx','lxx','lxxx'
        print("Initializing a new model from scratch")
        model_args = get_model_args(model_size, 'encoder', tokenizer.vocab_size)
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        print(model)
        print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    # number of parameters: 5981472   6m
    # number of parameters: 25315840  25m
    # number of parameters: 84969216  85m
    # number of parameters: 160627104 160m
    # number of parameters: 1459981440 1.5b

if __name__ == '__main__':
    cal_params()
    # unitest_flash_attn()
    # unitest_kv_cache()
    # unitest_transfomer()
