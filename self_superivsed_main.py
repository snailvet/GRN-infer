###############################################################################
# imports

import numpy as np
import os
import pickle as pkl
import pdb
import random
import torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.arguments import parse_arguments
from src.data_loading import init_data
from src.Transformer import GeneMasker, GRNInferModel

# parse arguments
opt = parse_arguments()

###############################################################################
# Set seeds and device

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
# main function

def train_model(opt):

    input_all, gene_name, tf_names = init_data(opt)
    print('Finished data preprocessing')

    #########################################
    ### setup the Model ###
    #########################################
    
    # set up model
    model = GRNInferModel(
        input_dim = input_all.shape[-1], 
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
        input_all
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

        for X, in dataloader:

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
        adj.append(attn[:, :, i].cpu().numpy())
    
    adj = np.array(adj)
    pkl.dump([adj, loss_save], open(f'{opt.save_name}', 'wb'))

###############################################################################

if __name__ == '__main__':

    # make directory to store output
    try:
        os.mkdir(opt.save_name)
    except FileExistsError:
        print('Directory already exists. Carrying on ...')


    # get dataset name from args
    if len(opt.dataset) == 0:
        # opt.data_file = path of input scRNA-seq file
        dataset_name = opt.data_file.split('/')[0]
    else:
        dataset_name = opt.dataset

    # update save name to include dataset_name
    opt.save_name = os.path.join(opt.save_name, dataset_name)

    # train!
    train_model(opt)

    print("Train complete... Yay!")