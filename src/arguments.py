###############################################################################
# imports 

import argparse

###############################################################################

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type = float, default = 0.01, help = "scale hyperparameter for the loss regularisation term")
    parser.add_argument("--attention_dim", default = 64, type = int, metavar = "N", help = "dimensionality of the attention layer")
    parser.add_argument("--attention_heads", default = 4, type = int, metavar = "N", help = "number of attention heads")
    parser.add_argument("--batchsize", type = int, default = 32)
    parser.add_argument("--dataset", type = str, default = '')
    parser.add_argument('--data_file', default = "data/grn_data.npy", type = str, help = 'path of input scRNA-seq file.')
    parser.add_argument("--dropout", type = float, default = 0.0, help = "Dropout probability in the dropout layers of the model")
    parser.add_argument("--dorothea_grade", type = str, default = "A", help = 'the minimum allowable Dorothea letter grade for transcription factors')
    parser.add_argument("--embed_dim", default = 64, type = int, metavar = "N", help = "embedding dimension")
    parser.add_argument('--gn_file', default = "data/gene_names.csv", type = str, help = 'path of gene names for embedding file.')
    parser.add_argument("--ffn_embed_dim", default = 64, type = int, metavar = "N", help = "embedding dimension for both feed forward networks of the model")
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate.')
    parser.add_argument("--n_epochs", type = int, default = 200, help = "number of epochs of training")
    parser.add_argument("--number_report_score", type = int, default = 4, help = 'the minimum allowable Number Report score for transcription factors')
    parser.add_argument("--PIDC_file", type = str, default = '')
    parser.add_argument("--save_name", type = str, default = './pretrain_output/')
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument('--tf_file', default = "data/transcription_factors.csv", type = str, help = 'path of transcription factor csv file.')
    parser.add_argument("--top_k", type = int, default = 20, help = "The number of largest attention weights to retain when calculating loss") 

    opt = parser.parse_args()

    return opt