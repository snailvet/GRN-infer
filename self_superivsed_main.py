# imports

import argparse
import numpy as np
import os
import pandas as pd
import pdb
import pickle as pkl
import random
import torch

from src.Transformer import retrieve_tf_names, GeneMasker, GRNInferModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

###############################################################################
# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type = float, default = 0.01, help = "scale hyperparameter for the loss regularisation term")
parser.add_argument("--attention_dim", default = 64, type = int, metavar = "N", help = "dimensionality of the attention layer")
parser.add_argument("--attention_heads", default = 4, type = int, metavar = "N", help = "number of attention heads")
parser.add_argument("--batchsize", type = int, default = 32)
parser.add_argument("--dataset", type = str, default = '')
parser.add_argument('--data_file', type = str, help = 'path of input scRNA-seq file.')
parser.add_argument("--dropout", type = float, default = 0.0, help = "Dropout probability in the dropout layers of the model")
parser.add_argument("--dorothea_grade", type = str, default = "A", help = 'the minimum allowable Dorothea letter grade for transcription factors')
parser.add_argument("--embed_dim", default = 64, type = int, metavar = "N", help = "embedding dimension")
parser.add_argument("--ffn_embed_dim", default = 64, type = int, metavar = "N", help = "embedding dimension for both feed forward networks of the model")
parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate.')
parser.add_argument("--n_epochs", type = int, default = 1500, help = "number of epochs of training")
parser.add_argument("--number_report_score", type = int, default = 4, help = 'the minimum allowable Number Report score for transcription factors')
parser.add_argument("--PIDC_file", type = str, default = '')
parser.add_argument("--save_name", type = str, default = './pretrain_output/')
parser.add_argument("--seed", type = int, default = 0)
parser.add_argument('--tf_data_file', type = str, help = 'path of transcription factor csv file.')
parser.add_argument("--top_k", type = int, default = 20, help = "The number of largest attention weights to retain when calculating loss") 
opt = parser.parse_args()

###############################################################################
# Set seeds 

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Using device: ", device)

###############################################################################

def init_data(opt):

    # load data
    data = pd.read_csv(opt.data_file, header = 0, index_col = 0).T
    data_values = data.values

    # build masks for data (nonzero values are true)
    d_mask_np = (data_values != 0).astype(float)
    d_mask = torch.FloatTensor(d_mask_np)

    # get mean and std for each column (excluding zero entries)
    means = []
    stds = []
    # loop through each column
    for i in range(data_values.shape[1]):
        tmp = data_values[:, i]

        # is there any non-zero entries in this column
        if sum(tmp != 0) == 0:
            # every entry is zero
            means.append(0)
            stds.append(1)
            
        else:
            # append mean and std of nonzero entries
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())

    # convert to np.array
    means = np.array(means)
    stds = np.array(stds)

    # set any Nan or inf entries of means to 1
    means[np.isnan(stds)] = 0
    means[np.isinf(stds)] = 0

    # set any Nan, inf, or zero entries of stds to 1
    stds[(np.isnan(stds))] = 1
    stds[np.isinf(stds)] = 1
    stds[stds == 0] = 1
    
    # shift each column by it's mean and scale it by it's std
    data_values = (data_values - means) / (stds)

    # Set any nan or inf entries to 0
    data_values[np.isnan(data_values)] = 0
    data_values[np.isinf(data_values)] = 0

    # Set any entries less then -20 to -20
    data_values = np.maximum(data_values, -20)
    # Set any entries greater then 20 to 20
    data_values = np.minimum(data_values, 20)

    # store as dataframe and float tensor
    data = pd.DataFrame(data_values, index = data.index, columns = data.columns)
    feat_train = torch.FloatTensor(data.values)

    return feat_train, d_mask_np, d_mask, data.columns, means, stds


def train_model(opt):

    input_all, d_mask_np, d_mask, gene_name, means, stds = init_data(opt)
    tf_data = pd.read_csv(opt.tf_data_file)
    tf_names = retrieve_tf_names(
        gene_name, 
        tf_data, 
        opt.dorothea_grade, 
        opt.number_report_score
    )
    print('Finished data preprocessing')

    #########################################
    ### setup the Model ###
    #########################################


    # catch if we have an embedding of dimension 1
    # we want a shape of [# cells, # genes, embedding dim]
    if input_all.dim() == 2:
        # the embedding for each gene is 1
        input_all = input_all.unsqueeze(-1)
        model_input_dim = 1
    else:
        model_input_dim = input_all.shape[-1]
    
    # set up model
    model = GRNInferModel(
        input_dim = model_input_dim, 
        attn_dim = opt.attention_dim,
        num_heads = opt.attention_heads, 
        pre_ffn_embed_dim = opt.ffn_embed_dim,
        post_ffn_embed_dim = opt.ffn_embed_dim,
        alpha = opt.alpha,
        top_k = opt.top_k,
        dropout = opt.dropout, 
    ).to(device)
    
    #setup masking object              
    g_masker = GeneMasker(
        genes = gene_name,
        t_factors = tf_names
    )

    # setup optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = opt.lr, 
        betas=(0.9, 0.999), 
        weight_decay = 0.01
    )
    
    #setup data loader
    dataset = TensorDataset(
        input_all, 
        d_mask, 
        torch.LongTensor(list(range(len(input_all))))
    )
    
    drop_last = True
    if len(input_all) < opt.batchsize:
        drop_last = False

    dataloader = DataLoader(
        dataset, 
        batch_size = opt.batchsize, 
        shuffle = True, 
        num_workers = 1, 
        drop_last = drop_last
    )
    
    # Train
    model.train()

    loss_save = []
    t = tqdm(range(opt.n_epochs))
    for epoch in t: #tqdm(range(opt.n_epochs)):
        loss_all = []
        model = model.to(device)

        for X, mask, idn in dataloader:

            optimizer.zero_grad()
            X_clone = X.clone()
            X = X.to(device)

            # mask non-transcription factor genes in input X
            g_masker.mask_non_tfs(X)
                                    
            # forward pass
            output, attn = model(X, return_attn = True)

            # calculate loss          
            # mask the transcription factors for the original input to the model and the models output
            g_masker.mask_tfs(output)
            g_masker.mask_tfs(X_clone)

            loss = model.loss(
                output, 
                X_clone.to(device), 
                attn 
            )

            # backward step
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.001)
            optimizer.step()

            # record loss
            loss_all.append(loss)

        # update tqdm ticker
        t.set_description(f'Finished training epoch: {epoch}; loss: {torch.stack(loss_all).mean()}')

        # 
        loss_save.append(torch.stack(loss_all).mean().cpu().item())
    
    print('Begin generate candidates GRN features')
    test_device = device 

    model = model.to(test_device)
    input = input_all.clone().to(test_device).unsqueeze(0)
    
    with torch.no_grad():
        output, attn = model(input, return_attn = True)
    
    adj = []
    for i in range(attn.shape[-1]): 
        adj.append(attn[:, :, i].cup().numpy())
    
    adj = np.array(adj)
    pkl.dump([adj, loss_save], open(f'{opt.save_name}', 'wb'))

if __name__ == '__main__':

    # make directory to store output
    try:
        os.mkdir(opt.save_name)
    except FileExistsError:
        print('Directory already exists. Carrying on ...')


    # get dataset name from args
    if len(opt.dataset) == 0:
        # opt.data_file = path of input scRNA-seq file
        dataset_name = [x for x in opt.data_file.split('/') if '00' in x][0]
    else:
        dataset_name = opt.dataset

    # update save name to include dataset_name
    opt.save_name = os.path.join(opt.save_name, dataset_name)

    # train!
    train_model(opt)

    print("Train complete... Yay!")