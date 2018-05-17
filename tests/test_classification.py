#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library

from pyHSICLasso import HSICLasso

standard_library.install_aliases()


class ClassificationTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_classification(self):
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

if __name__ == "__main__":
    unittest.main()
