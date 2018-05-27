#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

from future import standard_library

from pyHSICLasso import input_csv_file, input_matlab_file, input_tsv_file

standard_library.install_aliases()



class InputDataTest(unittest.TestCase):
    def test_input_csv(self):
        X_in, Y_in, featname = input_csv_file("./tests/test_data/csv_data.csv")
        X_in_row, X_in_col = X_in.shape
        Y_in_row, Y_in_col = Y_in.shape
        self.assertEqual(X_in_row, 2000)
        self.assertEqual(X_in_col, 62)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 62)

    def test_input_tsv(self):
        X_in, Y_in, featname = input_tsv_file("./tests/test_data/tsv_data.tsv")
        X_in_row, X_in_col = X_in.shape
        Y_in_row, Y_in_col = Y_in.shape
        self.assertEqual(X_in_row, 2000)
        self.assertEqual(X_in_col, 62)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 62)

    def test_input_matlab(self):
        X_in, Y_in, featname = input_matlab_file("./tests/test_data/matlab_data.mat")
        X_in_row, X_in_col = X_in.shape
        Y_in_row, Y_in_col = Y_in.shape
        self.assertEqual(X_in_row, 2000)
        self.assertEqual(X_in_col, 100)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 100)


if __name__ == "__main__":
    unittest.main()
