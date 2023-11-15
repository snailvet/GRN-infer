###############################################################################
# imports 

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Scaled dot product

def scaled_dot_product(Q, K, V):
    """
    Input:
        Q, K, V:    matrix vector products of the q, k, and v weight matrices 
                    and a gene embedding g (In LaTeX: Q = W_q\vec{g})
    """

    # Q, K, V should all be T x D_qkv matrices
    d_kqy = Q.size()[-1]

    # s_{ij}
    S = torch.matmul(Q, K.transpose(-2, -1))
    S = S / math.sqrt(d_kqy)

    # a_{ij}
    A = F.softmax(S, dim = -1)

    # h_{ij}
    H = torch.matmul(A, V)

    return H, A

###############################################################################
# Mutlihead Attention module

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        """
        Inputs:
            input_dim:  Dimensionality of the input
            embed_dim:  Dimensionality of the embedding that is output
            num_heads:  Number of heads to use in the attention block
        """

        super().__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.QKV_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.O_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention = False):

        batch_size, seq_length, _ = x.size()

        QKV = self.QKV_proj(x)

        # Separate Q, K, V from linear input
        QKV = QKV.reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        )
        QKV = QKV.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        Q, K, V = QKV.chunk(3, dim = -1)

        # Determine value outputs
        values, attention = scaled_dot_product(Q, K, V)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        O = self.O_proj(values)

        if return_attention:
            return O, attention
        else:
            return O

###############################################################################
# Transformer module

class ATTNLoss(nn.Module):

    def __init__(self, alpha, top_k):
        """
        Inputs:
            alpha:  regularisation weight hyper-parameter
            top_k:  the number of largest attention weights to keep when 
                    calculating the approximate attention matrix 
        """
        super().__init__()

        self.alpha = alpha #? Default value for alpha?
        self.top_k = top_k #? Default value for top_k?

        self.mse_loss = nn.MSELoss()

    def approx_attn(self, attn):

        tK = attn.topk(self.top_k, -1) # get top_k value from each 'row'
        aprx_attn = torch.zeros_like(attn)
        # replace top_k values back where they came from. All other elements zero 
        aprx_attn.scatter_(1, tK.indices, tK.values) 

        return aprx_attn

    def forward(self, x, y, attn):

        rec_loss = self.mse_loss(x, y)
        attn_loss = self.mse_loss(attn, self.approx_attn(attn))

        return rec_loss + self.alpha * attn_loss


###############################################################################
# Transformer module

class Transformer(nn.Module):

    def __init__(self, input_dim, attn_dim, num_heads, ffn_embed_dim, alpha, top_k, dropout = 0.0):
        """
        Inputs:
            input_dim:      Dimensionality of the input, will also be used for 
                            the dimensionality of the output
            attn_dim:       
            num_heads:      Number of heads to use in the attention block
            ffn_embed_dim:  Dimensionality of the hidden layer in the feed forward network
            alpha:          regularisation weight hyper-parameter for loss function
            top_k:          the number of largest attention weights to keep 
                            when calculating the approximate attention matrix 
            dropout:        Dropout probability to use in the dropout layers
        """
        super().__init__()

        # save parameters
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.dropout = dropout

        # Attention layer
        self.self_attn = MultiheadAttention(attn_dim, attn_dim, num_heads)

        # Pre Attention Feed Forward Network
        self.pre_ff_network = nn.Sequential(
            nn.Linear(input_dim, ffn_embed_dim),
            nn.Dropout(dropout), 
            nn.ReLU(inplace = True), 
            nn.Linear(ffn_embed_dim, attn_dim)
        )
        # Post Attention Feed Forward Network
        self.post_ff_network = nn.Sequential(
            nn.Linear(attn_dim, ffn_embed_dim),
            nn.Dropout(dropout), 
            nn.ReLU(inplace = True), 
            nn.Linear(ffn_embed_dim, input_dim)
        )

        # additional layers
        self.norm_1 = nn.LayerNorm(attn_dim)
        self.norm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        # Loss function
        self.loss = ATTNLoss(alpha, top_k)

    def forward(self, x, mask = None):

        # reshape to Batch x numgenes(seqlen) x 1
        x = x.unsqueeze(-1)

        x = self.pre_ff_network(x)

        # Attention part
        attn_out, attn = self.self_attn(x, mask = mask, return_attention = True)
        x = x + self.dropout(attn_out)
        x = self.norm_1(x)

        # FFN part
        ffn_out = self.post_ff_network(x)
        x = x + self.dropout(ffn_out) 
        x = self.norm_2(x)
        
        x = x.squeeze(-1)

        return x, attn
        
###############################################################################
# Masker

class TranscriptionFactorMasker():

    def __init__(self, genes, t_factors):
        """
        Inputs:
            genes:      a 1D array of all gene names
            t_factors:  a 1D array of all transcription factor names
        """

        self.genes = genes
        self.t_factors = t_factors

        # get indexes of transcription factors in genes
        self.tf_indexes = nn.nonzero(
            genes[:, None] == t_factors
        )[0]
        
    def build_mask(self, X):

        mask = nn.zeros_like(X)
        mask[:, self.tf_indexes] = 1

        return mask

###############################################################################
# Model

if __name__ == "__main__":

    pass

# Based by code from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html