from tiny_llama2.llama2 import load_model
from transformers import EsmTokenizer, EsmModel
def get_extractor(args):
    root = args.pretrained_lm_dir
    if args.model_scale == "s":
        model = 'mars_run-encoder-mars-s-train-val-d0.15-2023_10_07_22_45_11'
        # 5981472
    elif args.model_scale == "m":
        model = 'mars_run-encoder-mars-m-train-val-d0.15-2023_10_07_22_38_01'
    elif args.model_scale == "l":
        model = 'mars_run-encoder-mars-l-train-val-d0.15-2023_10_05_22_03_53'
        # 84969216
    elif args.model_scale == "lx":
        model = 'mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21'
    elif args.model_scale == "lxx":
        model = 'mars_run-encoder-mars-lxx-train-val-d0.15-2023_10_05_22_18_08'
    elif args.model_scale == "lxxx":
        model = 'mars_run-encoder-mars-lxxx-train-val-d0.15-2023_10_16_11_05_51'
    else:
        raise NotImplementedError
    
    ckpt_path = f'{root}/{model}/ckpt_175000.pt'
    extractor = load_model(ckpt_path, device='cpu')
    tokenizer = EsmTokenizer.from_pretrained("/home/bingxing2/ailab/scxlab0027/projects/ProbFold/vocab_esm_mars.txt")
    #  temporary model
    # model = load_model('/mnt/data/oss_beijing/wuhao/esm/mars_run-encoder-mars-s-train-val-d0.15-2023_10_07_22_45_11/ckpt_175000.pt')
    return extractor,tokenizer

def get_model_args(model_size, model_type, vocab_size, pretraine_mode='MLM', dropout = 0.0):
    """Get model args for a given model size and transformer type"""

    multiple_of = 32

    if model_size == "s":
        dim, n_layers, n_heads = 288,6,6
    elif model_size == "m":
        dim, n_layers, n_heads = 512,8,8
    elif model_size == "l":
        dim, n_layers, n_heads = 768,12,12
    elif model_size == "lx":
        dim, n_layers, n_heads = (768+288),12,12
    elif model_size == "lxx":
        dim, n_layers, n_heads = (1280),33,20
    elif model_size == "lxxx":
        dim, n_layers, n_heads = int(1280*1.5), 33, 20
    else:
        raise ValueError("Unknown model size")

    # model init
    model_args = dict(
        hidden_size=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        dropout=dropout,
        is_decoder=True if model_type == 'decoder' else False,
        pretrain_mode=pretraine_mode,
    )   # start with model_args from command line

    class MyObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    model_config = MyObject(model_args)

    return model_config