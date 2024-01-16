###############################################################################
# imports 

import math
import numpy as np
import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Mutlihead Attention module

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, top_k):
        """
        Inputs:
            input_dim:  Dimensionality of the input
            embed_dim:  Dimensionality of the embedding that is output
            num_heads:  Number of heads to use in the attention block
            top_k:      the number of largest attention weights to keep  
        """

        super().__init__()

        #? Why is this needed?
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k

        # Stack all weight matrices 1...h together for efficiency
        self.QKV_proj = nn.Linear(input_dim, 3 * embed_dim, bias = False)
        self.O_proj = nn.Linear(embed_dim, embed_dim, bias = False)

    def forward(self, x, return_attention = False):

        batch_size, n_genes, _ = x.size()

        QKV = self.QKV_proj(x)

        # Separate Q, K, V from linear input
        QKV = QKV.reshape(
            batch_size, n_genes, self.num_heads, 3 * self.head_dim
        )
        QKV = QKV.permute(0, 2, 1, 3) # [Batch, Head, NGenes, Dims]
        Q, K, V = QKV.chunk(3, dim = -1)

        # Determine value outputs
        values, attn = self.scaled_dot_product(Q, K, V)
        values = values.permute(0, 2, 1, 3) # [Batch, NGenes, Head, Dims]
        values = values.reshape(batch_size, n_genes, self.embed_dim)

        O = self.O_proj(values)

        if return_attention:
            return O, attn
        else:
            return O

    def scaled_dot_product(self, Q, K, V):
        """
        Input:
            Q, K, V:    matrix vector products of the q, k, and v weight matrices 
                        and a gene embedding g (eg. in LaTeX: Q = W_q\vec{g})
        """

        # Q, K, V should all be T x D_qkv matrices
        d_kqy = Q.size()[-1]

        # s_{ij}
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_kqy)

        # a_{ij}
        A = F.softmax(S, dim = -1)

        # retain top_k values of A
        tK = A.topk(self.top_k, -1)
        tkA = torch.zeros_like(A).scatter_(-1, tK.indices, tK.values)

        # h_{ij}
        H = torch.matmul(tkA, V)

        return H, A


###############################################################################
# Loss module

class ATTNLoss(nn.Module):

    def __init__(self, alpha, top_k):
        """
        Inputs:
            alpha:  regularisation weight hyper-parameter
            top_k:  the number of largest attention weights to keep when 
                    calculating the approximate attention matrix 
        """
        super().__init__()

        self.alpha = alpha 
        self.top_k = top_k 

        self.gene_mse_loss = nn.MSELoss()
        self.attn_mse_loss = nn.MSELoss()

    def approx_attn(self, attn):

        tK = attn.topk(self.top_k, -1) # get top_k value from each 'row'

        # replace top_k values back where they came from. All other elements zero 
        aprx_attn = torch.zeros_like(attn)
        aprx_attn.scatter_(-1, tK.indices, tK.values) 

        return aprx_attn

    def forward(self, x, y, attn):

        rec_loss = self.gene_mse_loss(x, y)
        attn_loss = self.attn_mse_loss(attn, self.approx_attn(attn))

        return rec_loss + self.alpha * attn_loss


###############################################################################
# GRNInferModel module

class GRNInferModel(nn.Module):

    def __init__(self, input_dim, attn_dim, num_heads, pre_ffn_embed_dim, post_ffn_embed_dim, alpha, top_k, dropout = 0.0):
        """
        Inputs:
            input_dim:          Dimensionality of the input, will also be used for, the dimensionality of the output
            attn_dim:           Dimensionality of the attention weight matrices
            num_heads:          Number of heads to use in the attention block
            pre_ffn_embed_dim:  Dimensionality of the hidden layers in the pre feed forward network
            post_ffn_embed_dim: Dimensionality of the hidden layers in the post feed forward network
            alpha:              regularisation weight hyper-parameter for loss function
            top_k:              the number of largest attention weights to keep when calculating the approximate attention matrix 
            dropout:            Dropout probability to use in the dropout layers
        """
        super().__init__()

        # save parameters
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.pre_ffn_embed_dim = pre_ffn_embed_dim
        self.post_ffn_embed_dim = post_ffn_embed_dim
        self.dropout = dropout

        # Attention layer
        self.self_attn = MultiheadAttention(attn_dim, attn_dim, num_heads, top_k)

        # Pre Attention Feed Forward Network
        self.pre_ff_network = nn.Sequential(
            nn.Linear(input_dim, pre_ffn_embed_dim),
            nn.Dropout(dropout), 
            nn.ReLU(inplace = True), 
            nn.Linear(pre_ffn_embed_dim, attn_dim)
        )
        # Post Attention Feed Forward Network
        self.post_ff_network = nn.Sequential(
            nn.Linear(attn_dim, post_ffn_embed_dim),
            nn.Dropout(dropout), 
            nn.ReLU(inplace = True), 
            nn.Linear(post_ffn_embed_dim, input_dim)
        )

        # additional layers
        self.norm_1 = nn.LayerNorm(attn_dim)
        self.norm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        # Loss function
        self.loss = ATTNLoss(alpha, top_k)

    def forward(self, x, return_attn = True):

        # pre FFN part
        x = self.pre_ff_network(x)

        # Attention part
        attn_out, attn = self.self_attn(x, return_attn)
        
        x = x + self.dropout(attn_out)
        x = self.norm_1(x)

        # post FFN part
        ffn_out = self.post_ff_network(x)
        x = ffn_out
        # x = x + self.dropout(ffn_out) 
        #? still creates an error x will have the attn_dim size
        x = self.norm_2(x)

        
        return x, attn
        
###############################################################################
# Masker

class GeneMasker():

    def __init__(self, genes, t_factors):
        """
        Inputs:
            genes:      a 1D array of all gene names 
            t_factors:  a 1D array of all transcription factor names
        """

        self.genes = genes
        self.t_factors = t_factors

        # get indexes of transcription factors in genes
        bools = np.isin(genes, t_factors)
        self.tf_indexes = np.array(np.where(bools))
        self.non_tf_indexes = np.array(np.where(bools == False))
        
    def mask_tfs(self, X, inplace = True):
        if not inplace:
            X = X.clone()

        X[:, self.tf_indexes, :] = 0

        if not inplace:
            return X

    def mask_non_tfs(self, X, inplace = True):
        if not inplace:
            X = X.clone()

        X[:, self.non_tf_indexes, :] = 0

        if not inplace:
            return X
    
###############################################################################
# 

if __name__ == "__main__":

    pass

# Based by code from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html