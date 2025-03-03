import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from utils.lm import get_extractor
from utils.tools import load_data, get_posbias
from utils.RNAformer.model.Riboformer_outfirst import RiboFormer
from utils.configuration import Config
from collections import OrderedDict
from tqdm import tqdm

def collate_fn(batch, scale):
    seqs, cts, _ = zip(*batch)
    max_len = max([len(seq)+2 for seq in seqs])
    data_dict = tokenizer.batch_encode_plus(seqs, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    data_dict['pos_bias'] = get_posbias(seqs, max_len, scale)
    ct_masks = [np.ones(ct.shape) for ct in cts]
    cts = [np.pad(ct, (0, max_len-ct.shape[0]), 'constant') for ct in cts]
    ## padding ct_mask
    ct_masks = [np.pad(ct_mask, (0, max_len-ct_mask.shape[0]), 'constant') for ct_mask in ct_masks]
    data_dict['ct'] = torch.FloatTensor(cts)
    data_dict['ct_mask'] = torch.FloatTensor(ct_masks)
    data_dict['seq_len'] = torch.tensor([len(seq) for seq in seqs])

    return data_dict

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_config = Config(config_file=args.config)
    model = RiboFormer(model_config['RNAformer'], extractor, args.is_freeze)
    
    # Load the saved model
    checkpoint = torch.load(args.model_path)
    original_state_dict = checkpoint['model_state_dict']

    new_state_dict = OrderedDict()
    for key, value in original_state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # "module."
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    _, _, test_dataset = load_data(args, None, aug=None, smooth=args.alpha)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=partial(collate_fn, scale=args.scale))

    criterion = nn.BCEWithLogitsLoss()

    
    def evaluate_threshold(model, test_loader, criterion, device):
        threshold = 0.45
        results = {'recall': [], 'precision': [], 'f1': []}

        with torch.no_grad():
            model.eval()
            model.to(device)
            for data_dict in tqdm(test_loader):
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(device)

                logits = model(data_dict)
                labels = data_dict['ct']
                loss_list = []
                bs = logits.shape[0]
                for idx in range(bs):
                    seq_length = data_dict['attention_mask'][idx].sum().item()
                    logit = logits[idx, :seq_length, :seq_length]
                    label = labels[idx, :logit.shape[0], :logit.shape[1]]
                    loss_list.append(criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))
                    
                    probs = torch.sigmoid(logit).detach().cpu().numpy()
                    label_np = label.detach().cpu().numpy().reshape(-1)

                    pred = (probs > threshold).astype(np.float32).reshape(-1)
                    results['recall'].append(
                        recall_score(label_np, pred))
                    results['precision'].append(
                        precision_score(label_np, pred))
                    results['f1'].append(
                        f1_score(label_np, pred))
                    # Clean GPU memory for this sample
                    del logit, label
                
                # Clean GPU memory after processing batch
                del logits, labels, data_dict
                torch.cuda.empty_cache()

        final_results = {
            'recall': np.mean(results['recall']),
            'precision': np.mean(results['precision']), 
            'f1': np.mean(results['f1'])
        }

        print('Final results: '
            f'precision: {final_results["precision"]:.6f}, '
            f'recall: {final_results["recall"]:.6f}, '
            f'F1: {final_results["f1"]:.6f}')
        
        # Clean up GPU memory
        model.cpu()
        torch.cuda.empty_cache()

    evaluate_threshold(model, test_loader, criterion, torch.device('cuda'))
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_scale', type=str, default='lx')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument('--config_dir', type=str, default='./config/t5')
    parser.add_argument("--pretrained_lm_dir", type=str, default='./pretrained')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--config', type=str, default='./utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--select', type=float, default=0)
    parser.add_argument('--replace', type=float, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint')

    args = parser.parse_args()

    extractor, tokenizer = get_extractor(args)

    main(args)