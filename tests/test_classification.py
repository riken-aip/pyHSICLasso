#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library
import numpy as np

from pyHSICLasso import HSICLasso

standard_library.install_aliases()


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_classification(self):

        np.random.seed(0)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.classification()

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.classification(5)
        self.assertEqual(self.hsic_lasso.A, [99, 1099, 199, 1181, 112])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.classification(10)
        self.assertEqual(self.hsic_lasso.A, [99, 1099, 199, 1181, 112,
                                             663, 761, 1869, 719, 977])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(5)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1581, 764])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(10)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1581, 764,
                                             1670, 1771, 896, 779, 1472])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(5, B)
        self.assertEqual(self.hsic_lasso.A, [99, 1099, 761, 250, 1199])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(10, B)
        self.assertEqual(self.hsic_lasso.A, [99, 1274, 183, 112, 199,
                                             390, 1181, 1099, 1339, 761])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(5, B)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 764, 1670, 248])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(10, B)
        self.assertEqual(self.hsic_lasso.A, [764, 1422, 779, 1771, 1581,
                                             248, 512, 1670, 896, 1136])

        # no error: exact divisors of n = 62
        self.hsic_lasso.classification(5, 2)
        self.hsic_lasso.classification(5, 31)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.classification(5, 3)

if __name__ == "__main__":
    unittest.main()
