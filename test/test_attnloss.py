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
        
        # generate input. Elements will be random values in [0, high]
        input = torch.randperm(high)[:heads * n * n].reshape(1, heads, n, n)
        temp = input.clone()

        # expected output. First initialise zero matrix
        expected_output = torch.zeros_like(input)
        for i in range(top_k):
            # get the max in each row
            maxs = temp.max(-1, keepdim = True)

            # save max to relevant position in expected output
            expected_output.scatter_(-1, maxs.indices, maxs.values)

            # set the max value in each row to -1 so we don't pick it again on the next loop
            temp.scatter_(-1, maxs.indices, -1)

        output = loss.approx_attn(input)

        self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":

    unittest.main()