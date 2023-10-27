import argparse
import os
import pickle as pkl
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training")
parser.add_argument("--layers", default=5, type=int, metavar="N", help="number of layers")
parser.add_argument("--embed_dim", default=64, type=int, metavar="N", help="embedding dimension")
parser.add_argument("--ffn_embed_dim", default=64, type=int, metavar="N", help="embedding dimension for FFN")
parser.add_argument("--attention_heads", default=4, type=int, metavar="N", help="number of attention heads")
parser.add_argument('--data_file', type=str, help='path of input scRNA-seq file.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument("--save_name", type=str, default='./pretrain_output/')
parser.add_argument("--dataset", type=str, default='')
parser.add_argument("--PIDC_file", type=str, default='')
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p", type=float, default=0.15,help='the fraction of random mask')
opt = parser.parse_args()
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
device = torch.device('cuda')


def init_data(opt):
    data = pd.read_csv(opt.data_file, header=0, index_col=0).T
    data_values = data.values
    d_mask_np = (data_values != 0).astype(float)
    d_mask = torch.FloatTensor(d_mask_np)
    means = []
    stds = []
    for i in range(data_values.shape[1]):
        tmp = data_values[:, i]
        if sum(tmp != 0) == 0:
            means.append(0)
            stds.append(1)
        else:
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())

    means = np.array(means)
    stds = np.array(stds)
    stds[np.isnan(stds)] = 1
    stds[np.isinf(stds)] = 1
    means[np.isnan(stds)] = 0
    means[np.isinf(stds)] = 0
    stds[stds == 0] = 1
    data_values = (data_values - means) / (stds)
    data_values[np.isnan(data_values)] = 0
    data_values[np.isinf(data_values)] = 0
    data_values = np.maximum(data_values, -20)
    data_values = np.minimum(data_values, 20)
    data = pd.DataFrame(data_values, index=data.index, columns=data.columns)
    feat_train = torch.FloatTensor(data.values)
    return feat_train, d_mask_np, d_mask, data.columns, means, stds


def train_model(opt):
    input_all, d_mask_np, d_mask, gene_name, means, stds = init_data(opt)
    print('finish data preprocessing')

    #########################################
    ### TODO 1: Implement the Transformer ###
    #########################################
    model = Transformer(input_dim=None, model_dim=None, num_heads=None,  d_model=None,
                                    dropout=None, top_k=None).to(device)
    #########################################
    
    n_gene = len(gene_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.01)
    dataset = TensorDataset(input_all, d_mask, torch.LongTensor(list(range(len(input_all)))))
    if len(input_all) < opt.batchsize:
        dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=1, drop_last=False)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=1, drop_last=True)
    model.train()
    loss_save = []
    for epoch in tqdm(range(opt.n_epochs)):
        loss_all = []
        model = model.to(device)
        for data, mask, idn in dataloader:
            data = torch.stack([data] * 1)
            mask = torch.stack([mask] * 1)
            optimizer.zero_grad()
            data_output = data.clone()
            data = data.to(device)

            
            ############################
            # TODO 2: design the mask ##
            ############################
            while True:
                mask_id = np.array(random.choices(range(data.shape[1] * data.shape[2]), k=int(data.shape[1] * data.shape[2] *opt.p)))
                data[0, mask_id // n_gene, mask_id % n_gene] = 0
                mask_new = torch.zeros_like(mask)
                mask_new[0, mask_id // n_gene, mask_id % n_gene] = 1
                mask_new = (mask_new * mask)
                if (mask_new).sum().item()>0:
                    break
            ############################
            mask_new = mask_new.to(device)
            zeros = (data == 0).float()
            output = model(data, zeros, return_attn=False)
            mask_new = mask_new.to(device)


            ############################
            # TODO 3: design the loss ##
            ############################
            loss = model.loss(output['logits'][:, :, :, 0], data_output.to(device), mask_new)
            ############################

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.001)
            optimizer.step()
            loss_all.append(loss)
        print('finish training epoch:',epoch, 'loss:',torch.stack(loss_all).mean())
        loss_save.append(torch.stack(loss_all).mean().cpu().item())
    print('begin generate candidates GRN features')
    test_device = torch.device('cuda')

    model = model.to(test_device)
    input = input_all.clone().to(test_device).unsqueeze(0)
    zeros = (input == 0).float()
    with torch.no_grad():
        output = model(input, zeros, return_attn=True)
    adj = []
    for i in range(output['row_attentions'].shape[-1]):
        adj.append(output['row_attentions'][:, :, i])
    adj = np.array(adj)
    pkl.dump([adj, loss_save], open(f'{opt.save_name}', 'wb'))


if __name__ == '__main__':
    try:
        os.mkdir(opt.save_name)
    except:
        print('dir exist')
    if len(opt.dataset) == 0:
        dataset_name = [x for x in opt.data_file.split('/') if '00' in x][0]
    else:
        dataset_name = opt.dataset

    opt.save_name = f'{opt.save_name}/{dataset_name}'
    train_model(opt)