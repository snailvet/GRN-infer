###############################################################################
# imports 

import numpy as np
import pandas as pd
import torch

###############################################################################
# initialise data for training 

def init_data(opt):

    # load embedding data 
    embedding_data = np.load(opt.data_file)
    embedding_data = np.transpose(embedding_data, (1, 0, 2))
    embedding_data = torch.FloatTensor(embedding_data)

    # catch if we have an embedding of dimension 1
    # we want a shape of [# cells, # genes, embedding dim]
    if embedding_data.dim() == 2:
        # the embedding for each gene is 1
        embedding_data = embedding_data.unsqueeze(-1)        

    # pdb.set_trace()
    # load gene name data 
    g_names = pd.read_csv(opt.gn_file)["g_names"]

    # load transcription factors 
    tf_data = pd.read_csv(opt.tf_file)
    tf_names = retrieve_tf_names(
        g_names,
        tf_data,
        opt.dorothea_grade,
        opt.number_report_score
    )

    return embedding_data, g_names, tf_names

###############################################################################
# transcription factor name retrieval 


def retrieve_tf_names(gene_names, tf_data, d_grade = None, nr_score = None):
    """
    Inputs:
        gene_names: the gene names from the scRNA-seq data
        tf_data:    the known transcription factor data ( must have columns GeneName, Dorothea, & Number_Report)
        d_grade:    the minimum allowable Dorothea letter grade (eg. "B" will select transcription factors with Dorothea grades of "A" and "B")
        nr_score:   the minimum allowable Number Report score (eg. "3" will allow any transcription factor with a number report of at least 3)
    """

    # find transcription factors that we have in our gene_names
    tf_data = tf_data[tf_data["GeneName"].isin(gene_names)] 
    
    # filter out tfs that have a lower d_grade (only if a value is provided)
    if d_grade:
        tf_data = tf_data[tf_data.Dorothea <= d_grade.upper()]

    # filter out tfs that have a lower nr_score (only if a value is provided)
    if nr_score:
        tf_data = tf_data[tf_data.Number_Report >= int(nr_score)]

    if tf_data.shape[0] == 0:
        raise ValueError("No qualifying transcription factors found in scRNA-seq data. Try adjusting minimum Dorothea grade and Number Report score") 

    return tf_data.GeneName

