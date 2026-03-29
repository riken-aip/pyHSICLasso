#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library
import numpy as np
import warnings

from pyHSICLasso import HSICLasso

standard_library.install_aliases()

class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_classification(self):

        np.random.seed(0)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.classification()

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(5, B = 0, discrete_x = True, n_jobs = 1)
        self.assertEqual(self.hsic_lasso.A, [764, 1422, 512, 248, 1581])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(10, B = 0, discrete_x = True, n_jobs = 1)
        self.assertEqual(self.hsic_lasso.A, [764, 1422, 512, 248, 1581, 
                                             1670, 1771, 896, 779, 266])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(5, B, 10, discrete_x = True)
        self.assertEqual(self.hsic_lasso.A, [764, 1422, 512, 248, 266])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(10, B, 10, discrete_x = True)
        self.assertEqual(self.hsic_lasso.A, [764, 1422, 512, 248, 1670, 
                                             1581, 266, 896, 1771, 779])

        # use non-divisor as block size
        with warnings.catch_warnings(record=True) as w:
        
            self.hsic_lasso.input("./tests/test_data/csv_data.csv")
            B = int(self.hsic_lasso.X_in.shape[1]/2) - 1
            n = self.hsic_lasso.X_in.shape[1]
            numblocks = n / B
            
            self.hsic_lasso.classification(10, B, 10, discrete_x = True)
            self.assertEqual(self.hsic_lasso.A, [1422, 764, 512, 248, 1670, 
                                                 1581, 896, 266, 1771, 779])
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, RuntimeWarning)
            self.assertEqual(str(w[-1].message), "B {} must be an exact divisor of the \
number of samples {}. Number of blocks {} will be approximated to {}.".format(B, n, numblocks, int(numblocks)))

        # Covariates
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        covars = self.hsic_lasso.X_in[[1422, 512],:].T
        self.hsic_lasso.classification(5, B = 0, n_jobs = 1, covars = covars)
        self.assertEqual(self.hsic_lasso.A, [622, 841, 1636, 1891, 116])

if __name__ == "__main__":
    unittest.main()
