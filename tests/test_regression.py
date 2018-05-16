#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library

from pyHSICLasso import HSICLasso

standard_library.install_aliases()


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_regression(self):
        with self.assertRaises(UnboundLocalError):
            self.hsic_lasso.regression()

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(5)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 299, 1574, 1645])

        self.hsic_lasso.input("./tests/test_data/matlab_data.mat")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1099, 99, 299, 1574, 1645, 173,
                                             1299, 199, 90, 1473])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(5)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1670, 779])

        self.hsic_lasso.input("./tests/test_data/csv_data.csv")
        self.hsic_lasso.regression(10)
        self.assertEqual(self.hsic_lasso.A, [1422, 512, 248, 1670, 779, 244,
                                             1581, 764, 1771, 1380])


if __name__ == "__main__":
    unittest.main()
