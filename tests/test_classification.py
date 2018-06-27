#!/usr/bin.env python
# coding: utf-8

import unittest

import numpy as np

from pyHSICLasso import HSICLasso


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_classification(self):

        np.random.seed(0)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.classification()

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.classification(5)
        self.assertEqual(self.hsic_lasso.A, [99, 831, 1099, 467, 220])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.classification(10)
        self.assertEqual(self.hsic_lasso.A, [99, 831, 467, 1099, 220, 1109,
                                             1918, 694, 1126, 1001])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(5)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1670, 779])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.classification(10)
        self.assertEqual(self.hsic_lasso.A, [1422, 1670, 512, 248, 779, 1581,
                                             764, 244, 1771, 1380])

        # Blocks
        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(5, B)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 1291, 126, 1260])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(10, B)
        self.assertEqual(self.hsic_lasso.A, [468, 1099, 220, 831, 694,
                                             99, 618, 548, 1901, 1001])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(5, B)
        self.assertEqual(self.hsic_lasso.A, [1670, 1422, 512, 764, 248])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        B = int(self.hsic_lasso.X_in.shape[1]/2)
        self.hsic_lasso.classification(10, B)
        self.assertEqual(self.hsic_lasso.A, [779, 1670, 1422, 248, 764,
                                             1771, 512, 1136, 1581, 65])

        # no error: exact divisors of n = 62
        self.hsic_lasso.classification(5, 2)
        self.hsic_lasso.classification(5, 31)

        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.classification(5, 3)

if __name__ == "__main__":
    unittest.main()
