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
warnings.simplefilter('always')

class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_regression(self):

        np.random.seed(0)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.regression()

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(5, B = 0, n_jobs = 1)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 1299, 1477])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(10, B = 0, n_jobs = 1)
        self.assertEqual(self.hsic_lasso.A, [1099, 199, 99, 1299, 1477,
                                             1073, 1405, 1596, 375, 358])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(5, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 1299, 1477])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(10, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1099, 199, 99, 1299, 1477,
                                             1073, 1405, 1363, 375, 492])

        # use non-divisor as block size
        with warnings.catch_warnings(record=True) as w:
        
            self.hsic_lasso.input("./tests/test_data/csv_data.csv")
            B = int(self.hsic_lasso.X_in.shape[1]/2) - 1
            n = self.hsic_lasso.X_in.shape[1]
            numblocks = n / B
            
            self.hsic_lasso.regression(10, B, 10)
            self.assertEqual(self.hsic_lasso.A, [1422, 248, 512, 1670, 1581,
                                                 764, 896, 1771, 779, 1472])
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, RuntimeWarning)
            self.assertEqual(str(w[-1].message), "B {} must be an exact divisor of the \
number of samples {}. Number of blocks {} will be approximated to {}.".format(B, n, numblocks, int(numblocks)))

        # Covariates
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        covars = self.hsic_lasso.X_in[[99,299],:].T
        self.hsic_lasso.regression(5, B = 0, n_jobs = 1, covars = covars)
        self.assertEqual(self.hsic_lasso.A, [199, 1477, 1073, 1405, 1596])

if __name__ == "__main__":
    unittest.main()
