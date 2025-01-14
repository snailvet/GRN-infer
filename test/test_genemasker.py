###############################################################################
# Imports 

import numpy as np
import pdb
import random
import string
import torch
import unittest

from src.Transformer import GeneMasker

###############################################################################
# 

class TestGeneMasker(unittest.TestCase):

    def random_gene_name(self, length):
        letters = string.ascii_uppercase
        return "".join(random.choice(letters) for i in range(length))


    def test___init__(self):
        
        # setup
        num_genes = 20
        num_tfs = 5
        word_length = 5
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes)])
        
        r = np.arange(gene_names.shape[0])
        tf_indexes = np.random.choice(r, num_tfs, replace = False)
        non_tf_indexes = r[np.in1d(r, tf_indexes) == False]
        
        tf_names = gene_names[tf_indexes]
        g_masker = GeneMasker(gene_names, tf_names)


        self.assertTrue(np.array_equal(tf_indexes.sort(), g_masker.tf_indexes.sort()))
        self.assertTrue(np.array_equal(non_tf_indexes.sort(), g_masker.non_tf_indexes.sort()))

    def test_approx_attn(self):
        
        # setup
        num_genes = 5
        num_tfs = 1
        word_length = 5

        # generate random gene names
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes)])
        
        # randomly pick indexes that are tfs
        r = np.arange(gene_names.shape[0])
        tf_indexes = np.random.choice(r, num_tfs, replace = False)
        non_tf_indexes = r[np.in1d(r, tf_indexes) == False]
        tf_names = gene_names[tf_indexes]
        
        # setup GeneMasker object
        g_masker = GeneMasker(gene_names, tf_names)

        # generate a random test attention matrix
        high = 1000
        rows = 5
        embedding_dim = 2
        input = torch.randint(high, size = (rows, num_genes, embedding_dim))

        # expected outputs 
        expected_mask_tfs = input.clone()
        expected_mask_non_tfs = input.clone()
        expected_mask_tfs[:, tf_indexes, :] = 0
        expected_mask_non_tfs[:, non_tf_indexes, :] = 0

        # actual outputs
        mask_tfs = input.clone()
        mask_non_tfs = input.clone()
        g_masker.mask_tfs(mask_tfs)
        g_masker.mask_non_tfs(mask_non_tfs)

        # pdb.set_trace()
        self.assertTrue(torch.equal(mask_tfs, expected_mask_tfs))
        self.assertTrue(torch.equal(mask_non_tfs, expected_mask_non_tfs))
        

if __name__ == "__main__":

    unittest.main()