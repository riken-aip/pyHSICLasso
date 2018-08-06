#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library
import numpy as np

from pyHSICLasso import HSICLasso

standard_library.install_aliases()


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_regression(self):

        np.random.seed(0)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.regression()

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(5)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 1299, 299])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 1299, 1477,
                                             1405, 1073, 299, 1596, 358])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(5)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1581, 764])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1581, 764,
                                             1670, 1771, 896, 779, 1472])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(5, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 299, 1477])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(10, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 199, 1477, 299, 
                                             1405, 1073, 1299, 1596, 358])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(5, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 764, 1581])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(10, B, 10)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 764, 1581, 
                                             1670, 1771, 896, 779, 1413])

        # no error: exact divisors of n = 62
        self.hsic_lasso.regression(5, 2)
        self.hsic_lasso.regression(5, 31)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.regression(5, 3)

if __name__ == "__main__":
    unittest.main()
