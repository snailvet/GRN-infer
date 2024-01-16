###############################################################################
# Imports 

import os
import numpy as np
import pandas as pd
import random
import string
import unittest

from src.Transformer import retrieve_tf_names

###############################################################################
# 

class Test_retrieve_tf_names(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_file_path = "test/test.csv"
        self.columns = ["GeneName", "Dorothea", "Number_Report"]

    def random_gene_name(self, length):
        letters = string.ascii_uppercase
        return "".join(random.choice(letters) for i in range(length))
    
    def random_dorothea(self, letters):
        return random.choice(letters)
    
    def random_number_report(self, numbers):
        return random.choice(numbers)
    
    def build_csv(self):

        raise NotImplementedError

    def test_no_restrictions(self):

        num_genes = 10
        num_tfs = 5
        word_length = 5

        # build csv
        tf_names = np.array([self.random_gene_name(word_length) for i in range(num_tfs)])
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes - num_tfs)])
        gene_names = np.concatenate((gene_names, tf_names))
        
        dorothea = np.array([self.random_dorothea(["A", "B", "C", "D"]) for i in range(num_tfs)])
        
        number_report = np.array([self.random_number_report(range(1, 5)) for i in range(num_tfs)])
        
        df = pd.DataFrame.from_dict(dict(zip(self.columns, [tf_names, dorothea, number_report])))
        df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows

        output = retrieve_tf_names(gene_names, df)
        expected_output = tf_names

        self.assertTrue(np.array_equal(output.sort_values(), np.sort(expected_output)))
    
    def test_at_least_dorothea_B(self):

        num_genes = 10
        num_tfs = 5
        num_dorothea = 2
        word_length = 5

        # build csv
        expected_output = np.array([self.random_gene_name(word_length) for i in range(num_dorothea)])
        tf_names = np.concatenate((
            expected_output,
            np.array([self.random_gene_name(word_length) for i in range(num_tfs - num_dorothea)]))
        )
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes - num_tfs)])
        gene_names = np.concatenate((gene_names, tf_names))
        
        dorothea = np.concatenate((
            np.array([self.random_dorothea(["A", "B"]) for i in range(num_dorothea)]), 
            np.array([self.random_dorothea(["C", "D"]) for i in range(num_tfs - num_dorothea)]), 
        ))
        number_report = np.array([self.random_number_report(range(1, 5)) for i in range(num_tfs)])
        
        df = pd.DataFrame.from_dict(dict(zip(self.columns, [tf_names, dorothea, number_report])))
        df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows

        output = retrieve_tf_names(gene_names, df, "B")

        self.assertTrue(np.array_equal(output.sort_values(), np.sort(expected_output)))
    
    def test_at_least_number_report_3(self):

        num_genes = 10
        num_tfs = 5
        num_nr = 2
        word_length = 5

        # build csv
        expected_output = np.array([self.random_gene_name(word_length) for i in range(num_nr)])
        tf_names = np.concatenate((
            expected_output,
            np.array([self.random_gene_name(word_length) for i in range(num_tfs - num_nr)]))
        )
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes - num_tfs)])
        gene_names = np.concatenate((gene_names, tf_names))
        
        dorothea = np.array([self.random_dorothea(["A", "B", "C", "D"]) for i in range(num_tfs)])

        number_report = np.concatenate((
            np.array([self.random_number_report([3, 4]) for i in range(num_nr)]), 
            np.array([self.random_number_report([1, 2]) for i in range(num_tfs - num_nr)]), 
        ))
        
        df = pd.DataFrame.from_dict(dict(zip(self.columns, [tf_names, dorothea, number_report])))
        df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows

        output = retrieve_tf_names(gene_names, df, nr_score = 3)

        self.assertTrue(np.array_equal(output.sort_values(), np.sort(expected_output)))
    
    def test_at_least_dorothea_B_number_report_3(self):

        num_genes = 20
        num_tfs = 10
        num_both = 6
        num_dorothea = 2
        num_nr = 2
        word_length = 5

        # build csv
        expected_output = np.array([self.random_gene_name(word_length) for i in range(num_both)])
        tf_names = np.concatenate((
            expected_output,
            np.array([self.random_gene_name(word_length) for i in range(num_tfs - num_both)]))
        )
        gene_names = np.array([self.random_gene_name(word_length) for i in range(num_genes - num_tfs)])
        gene_names = np.concatenate((gene_names, tf_names))
        
        dorothea = np.concatenate((
            np.array([self.random_dorothea(["A", "B"]) for i in range(num_both + num_dorothea)]), 
            np.array([self.random_dorothea(["C", "D"]) for i in range(num_tfs - num_both - num_dorothea)]), 
        ))

        number_report = np.concatenate((
            np.array([self.random_number_report([3, 4]) for i in range(num_both)]), 
            np.array([self.random_number_report([1, 2]) for i in range(num_dorothea)]), 
            np.array([self.random_number_report([3, 4]) for i in range(num_nr)]), 
            np.array([self.random_number_report([1, 2]) for i in range(num_tfs - num_both - num_dorothea - num_nr)]), 
        ))
        
        df = pd.DataFrame.from_dict(dict(zip(self.columns, [tf_names, dorothea, number_report])))
        df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows

        output = retrieve_tf_names(gene_names, df, d_grade = "B", nr_score = 3)

        self.assertTrue(np.array_equal(output.sort_values(), np.sort(expected_output)))
    

if __name__ == "__main__":

    unittest.main()