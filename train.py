import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup
from utils.lm import get_extractor,get_model_args
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from utils.tools import load_data, get_posbias
from utils.RNAformer.model.Riboformer_outfirst import RiboFormer
from utils.predictor import Augmentation
from utils.configuration import Config


seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch, scale):
    seqs, cts, _ = zip(*batch)
    max_len = max([len(seq)+2 for seq in seqs])
    data_dict = tokenizer.batch_encode_plus(seqs, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    data_dict['pos_bias'] = get_posbias(seqs, max_len, scale)

    ## padding ct
    ct_masks = [np.ones(ct.shape) for ct in cts]
    cts = [np.pad(ct, (0, max_len-ct.shape[0]), 'constant') for ct in cts]
    ## padding ct_mask
    ct_masks = [np.pad(ct_mask, (0, max_len-ct_mask.shape[0]), 'constant') for ct_mask in ct_masks]
    data_dict['ct'] = torch.FloatTensor(cts)
    data_dict['ct_mask'] = torch.FloatTensor(ct_masks)
    data_dict['seq_len'] = torch.tensor([len(seq) for seq in seqs])

    return data_dict


def main(args) :
    name = f'[MARS_former_outfirst{args.scale}_s{args.select}_r{args.replace}_{args.c}]{args.mode}_' \
            f'{args.model_scale}_' \
            f'lr{args.lr}_' \
            f'bs{args.batch_size}_' \
            f'gs{args.gradient_accumulation_steps}' \
            f'epo{args.num_epochs}_' \
            f'warm{args.warmup_epos}epoch'

    if args.is_freeze:
        name += '_freeze'
    if args.trained_weight != None :
        name += '_fromtrained'

    args.pdb_dir = f'{args.data_dir}/PDB_SS'
    args.bprna_dir = f'{args.data_dir}/bpRNA'
    args.spotrna_dir = f'{args.data_dir}/PDB_spotrna'
    args.javapdb_dir = f'{args.data_dir}/java_PDB'

    if args.save:
        ckpt_path = os.path.join(args.ckpt_dir, name)
        os.makedirs(ckpt_path, exist_ok=True)

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with='wandb')

    model_config = get_model_args(args.model_scale, 'encoder', tokenizer.vocab_size)
    model_config = Config(config_file=args.config)
    model = RiboFormer(model_config['RNAformer'], extractor, args.is_freeze)

    print(accelerator.device)

    if args.trained_weight != None :
        pretrained_dict = torch.load(args.trained_weight+'/best_val/pytorch_model.bin')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)    

    augmentation = Augmentation(args.select, args.replace)
    train_dataset, val_dataset, test_dataset = load_data(args, None, aug=augmentation, smooth=args.alpha)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=partial(collate_fn, scale = args.scale))
    if len(val_dataset)!=0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_fn, scale = args.scale))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=partial(collate_fn, scale = args.scale))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## num_processes from accelerator
    per_steps_one_epoch = len(train_dataset) // args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    num_warmup_steps = per_steps_one_epoch * args.warmup_epos
    num_training_steps = per_steps_one_epoch * args.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    if accelerator.is_main_process:
        wandb.init(project='SS_prediction')
        wandb.run.name = name
        wandb.run.save()
        wandb.watch(model)
        print(name)

    criterion = nn.BCEWithLogitsLoss()

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    best_val, best_test = 0, 0

    firstload = True

    for epoch in range(args.restart, args.num_epochs):
        if args.restart != 0:
            checkpoint = torch.load(os.path.join(args.ckpt_dir, name+'.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            if accelerator.is_main_process and firstload:
                firstload = False
                for _ in range(args.restart // 5):
                    log_dict = {'lr': 0, 'train_loss': 0}
                    if len(val_dataset)!=0:
                        log_dict.update({'VL0/loss': 0, 'VL0/precision': 0, 'VL0/recall': 0, 'VL0/F1': 0})
                    log_dict.update({'TS0/loss': 0, 'TS0/precision': 0,
                                    'TS0/recall': 0, 'TS0/F1': 0})
                    wandb.log(log_dict)

        print(f'epoch {epoch}')
        model.train()
        start_time = time.time()
        for data_dict in tqdm(train_loader):
            with accelerator.accumulate(model):
                logits = model(data_dict)
                labels = data_dict['ct']
                loss_list = []
                bs = logits.shape[0]
                for idx in range(bs):
                    seq_length = data_dict['attention_mask'][idx].sum().item()
                    logit = logits[idx, :seq_length, :seq_length]
                    label = labels[idx, :logit.shape[0], :logit.shape[1]]
                    loss_list.append(criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))
                    del logit, label, seq_length
                    
                del logits, labels
                loss = torch.stack(loss_list).mean()
                accelerator.backward(loss)
                del loss_list
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()

            gather_loss = accelerator.gather(loss.detach().float()).mean().item()
            train_loss_list.append(gather_loss)

        if epoch % 5 == 4:
            val_recall_list, val_precision_list, val_f1_list = [], [], [], []
            test_recall_list, test_precision_list, test_f1_list = [], [], [], []

            threshold = 0.5
            if len(val_dataset)!=0:
                with torch.no_grad():
                    model.eval()
                    for data_dict in val_loader:
                        for key in data_dict:
                            data_dict[key] = data_dict[key].to(accelerator.device)
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
                            pred = (probs > threshold).astype(np.float32)
                            val_recall_list.append(recall_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                            val_precision_list.append(
                                precision_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                            val_f1_list.append(f1_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                            del logit, label, seq_length
                            
                        loss = torch.stack(loss_list).mean()
                        val_loss_list.append(loss.item())
                        torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                for data_dict in test_loader:
                    for key in data_dict:
                        data_dict[key] = data_dict[key].to(accelerator.device)

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
                        pred = (probs > threshold).astype(np.float32)
                        test_recall_list.append(recall_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                        test_precision_list.append(
                            precision_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                        test_f1_list.append(f1_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                        del logit, label
                        
                    loss = torch.stack(loss_list).mean()
                    test_loss_list.append(loss.item())
                    torch.cuda.empty_cache()

            if args.save:
                accelerator.wait_for_everyone()
                if best_val < np.mean(val_precision_list) + np.mean(val_recall_list) + np.mean(val_f1_list):
                    best_val = np.mean(val_precision_list) + np.mean(val_recall_list) + np.mean(val_f1_list)
                    if accelerator.is_main_process:
                        accelerator.save_state(f'{ckpt_path}/best_val')

                if best_test < np.mean(test_precision_list) + np.mean(test_recall_list) + np.mean(test_f1_list):
                    best_test = np.mean(test_precision_list) + np.mean(test_recall_list) + np.mean(test_f1_list)
                    if accelerator.is_main_process:
                        accelerator.save_state(f'{ckpt_path}/best_test')
                accelerator.wait_for_everyone()

            end_time = time.time()

            if accelerator.is_main_process:
                print(
                    f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.2f}')
                print(
                    f'[VL0] Loss: {np.mean(val_loss_list):.6f}, precision: {np.mean(val_precision_list):.6f}, recall: {np.mean(val_recall_list):.6f}, F1: {np.mean(val_f1_list):.6f}')
                print(
                    f'[TS0] loss: {np.mean(test_loss_list):.6f}, precision: {np.mean(test_precision_list):.6f}, recall: {np.mean(test_recall_list):.6f}, F1: {np.mean(test_f1_list):.6f}')
                log_dict = {'lr': optimizer.param_groups[0]["lr"], 'train_loss': np.mean(train_loss_list)}
                if len(val_dataset)!=0:
                    log_dict.update({'VL0/loss': np.mean(val_loss_list), 'VL0/precision': np.mean(val_precision_list), 'VL0/recall': np.mean(val_recall_list), 'VL0/F1': np.mean(val_f1_list)})
                log_dict.update({'TS0/loss': np.mean(test_loss_list), 'TS0/precision': np.mean(test_precision_list),
                                'TS0/recall': np.mean(test_recall_list), 'TS0/F1': np.mean(test_f1_list)})
                wandb.log(log_dict)
            torch.cuda.empty_cache()
            train_loss_list, val_loss_list, test_loss_list = [], [], []





if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epos', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--model_scale', type=str, default='lx')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument('--trained_weight',type=str)
    parser.add_argument('--config_dir',type=str,default='./config/t5')
    parser.add_argument('--pretrained_lm_dir', type=str, default='./model')
    parser.add_argument('--data_dir', default= './data')
    parser.add_argument('--ckpt_dir', default='./model/')
    parser.add_argument('--c', type=str, default='')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--config', type=str, default='./utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml')
    parser.add_argument('--scale', type=float, default=0.01)
    parser.add_argument('--K', type=float, default=1.7)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--select', type=float, default=1)
    parser.add_argument('--replace', type=float, default=0)
    parser.add_argument('--restart', type=int, default=0)

    args = parser.parse_args()


    ## get pretrained model
    extractor, tokenizer = get_extractor(args)

    main(args)
