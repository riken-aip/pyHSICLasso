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
        self.assertEqual(self.hsic_lasso.A, [1099, 299, 99, 1574, 1645])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1099, 1574, 299, 1645, 99, 173,
                                             1299, 199, 90, 1473])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(5)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1670, 779])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1422, 1670, 512, 248, 779, 1581,
                                             764, 244, 1771, 1380])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(5, B)
        self.assertEqual(self.hsic_lasso.A, [1099, 299, 173, 99, 480])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(10, B)
        self.assertEqual(self.hsic_lasso.A, [1099, 1645, 574, 116, 1378,
                                             299, 1116, 1574, 1875, 518])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(5, B)
        self.assertEqual(self.hsic_lasso.A, [1670, 1422, 512, 764, 248])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.regression(10, B)
        self.assertEqual(self.hsic_lasso.A, [779, 1670, 1422, 248, 764,
                                             1771, 512, 1136, 1581, 65])

        # no error: exact divisors of n = 62
        self.hsic_lasso.regression(5, 2)
        self.hsic_lasso.regression(5, 31)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.regression(5, 3)

if __name__ == "__main__":
    unittest.main()
