###############################################################################
# imports 

import argparse
import os

from src.Transformer import GRNInferModel
from torch.utils.data import DataLoader, TensorDataset

###############################################################################
# argument setup

def setup_args(root_dir, week_dir):
    opt = argparse.Namespace(
        alpha = 0.01, # scale hyperparameter for the loss regularisation term
        attention_dim = 64, # dimensionality of the attention layer
        attention_heads = 4, # number of attention heads
        batchsize = 32,
        dataset = "",
        data_file = os.path.join(root_dir, "data/grn_data.npy"), # path of input scRNA-seq embedding file
        dropout = 0.0, # Dropout probability in the dropout layers of the model
        dorothea_grade = "A", # the minimum allowable Dorothea letter grade for transcription factors
        embed_dim = 64, # embedding dimension
        gn_file = os.path.join(root_dir, "data/gene_names.csv"), # path of gene names for embedding file
        ffn_embed_dim = 64, # embedding dimension for both feed forward networks of the model
        lr = 1e-3, # learning rate
        n_epochs = 200, # number of epochs of training
        number_report_score = 4, # the minimum allowable Number Report score for transcription factors
        PIDC_file = os.path.join(week_dir, "PIDC_output_file.txt"),
        save_name = 'pretrain_output',
        seed = 0,
        tf_file = os.path.join(root_dir, "data/transcription_factors.csv"), # path of transcription factor csv file
        top_k = 20, # The number of largest attention weights to retain when calculating loss
    )

    return opt

def setup_supervised_args(root_dir, data_dir):
    opt = argparse.Namespace(
        split_file = "split_non_1.pkl",
        train_y_file = 'train_y_non.npy',
        data_dir = os.path.join(root_dir, data_dir, "inputs"),
        pre_GRN_file = os.path.join(root_dir, data_dir, "outputs", 'pretrain_output.pkl'),
        output_file = os.path.join(root_dir, data_dir, "outputs", 'performance.pkl'),
        tree_num = 10,
    )

    return opt

###############################################################################
# model etc setup

def setup_model(input_all, opt, device):

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

    return model

def setup_optimizer(model, optimiser_class, opt):
    optimizer = optimiser_class(
        model.parameters(),
        lr = opt.lr,
        weight_decay = 0.01
    )

    return optimizer

def setup_dataloader(input_all, opt):
    dataset = TensorDataset(input_all)

    drop_last = True
    if len(input_all) < opt.batchsize:
        drop_last = False

    dataloader = DataLoader(
        dataset,
        batch_size = opt.batchsize,
        shuffle = False,
        num_workers = 1,
        drop_last = drop_last
    )

    return dataloader
