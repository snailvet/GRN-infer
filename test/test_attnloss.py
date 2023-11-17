###############################################################################
# Imports 

import numpy as np
import torch
import unittest

from src.Transformer import ATTNLoss

###############################################################################
# 

class TestATTnLoss(unittest.TestCase):

    def test_approx_attn(self):
        
        # setup
        alpha = 0.1
        top_k = 2
        loss = ATTNLoss(alpha, top_k)

        n = 5
        heads = 4
        high = 1000
        
        # input = torch.randint(high, size = (1, heads, 5, 5))
        input = torch.randperm(high)[:heads * n * n].reshape(1, heads, n, n)
        temp = input.clone()
        expected_output = torch.zeros_like(input)
        
        for i in range(top_k):
            maxs = temp.max(-1, keepdim = True)
            expected_output.scatter_(-1, maxs.indices, maxs.values)
            temp.scatter_(-1, maxs.indices, -1)

        output = loss.approx_attn(input)

        self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":

    unittest.main()