from prifold.llama2 import load_model
from transformers import EsmTokenizer, EsmModel

def get_extractor(args):
    root = args.pretrained_lm_dir
    if args.model_scale == "6m":
        model = 'mars-6m'
    elif args.model_scale == "25m":
        model = 'mars-25m'
    elif args.model_scale == "85m":
        model = 'mars-85m'
    elif args.model_scale == "160m":
        model = 'mars-160m'
    else:
        raise NotImplementedError
    
    ckpt_path = f'{root}/{model}/ckpt_175000.pt'
    extractor = load_model(ckpt_path, device='cpu')
    tokenizer = EsmTokenizer.from_pretrained("vocab_esm_mars.txt")

    return extractor,tokenizer

def get_model_args(model_size, model_type, vocab_size, pretraine_mode='MLM', dropout = 0.0):
    """Get model args for a given model size and transformer type"""

    multiple_of = 32

    if model_size == "6m":
        dim, n_layers, n_heads = 288,6,6
    elif model_size == "25m":
        dim, n_layers, n_heads = 512,8,8
    elif model_size == "85m":
        dim, n_layers, n_heads = 768,12,12
    elif model_size == "160m":
        dim, n_layers, n_heads = 1056,12,12  # 768+288
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