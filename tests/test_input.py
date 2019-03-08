#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

import numpy as np
from future import standard_library

from pyHSICLasso import HSICLasso

standard_library.install_aliases()



class InputTest(unittest.TestCase):
    def setUp(self):
        self.hsic_lasso = HSICLasso()

    def test_check_arg(self):
        with self.assertRaises(SyntaxError):
            self.hsic_lasso._check_args([])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([1, 2, 3])
        with self.assertRaises(ValueError):
            self.hsic_lasso._check_args(["txt"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args(["hoge.txt"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args(["hogecsv"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([123])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([[1, 2, 3]])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([np.array([1, 2, 3])])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args(["hoge", "hoge"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args(["hoge", [1, 2, 3]])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([[1, 2, 3], "hoge"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args(["hoge", np.array([1, 2, 3])])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([np.array([1, 2, 3]), "hoge"])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([123, [1, 2, 3]])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([[1, 2, 3], 123])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([123, np.array([1, 2, 3])])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([np.array([1, 2, 3]), 123])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([[1, 2, 3], np.array([1, 2, 3])])
        with self.assertRaises(TypeError):
            self.hsic_lasso._check_args([np.array([1, 2, 3]), [1, 2, 3]])

        self.assertTrue(self.hsic_lasso._check_args(["hoge.csv"]))
        self.assertTrue(self.hsic_lasso._check_args(["hoge.tsv"]))
        self.assertTrue(self.hsic_lasso._check_args(["hoge.mat"]))
        self.assertTrue(self.hsic_lasso._check_args([np.array([1, 2, 3]),
                                                    np.array([1, 2, 3])]))
        self.assertTrue(self.hsic_lasso._check_args([[1, 2, 3], [1, 2, 3]]))

    def test_input_data_file(self):
        self.assertTrue("./tests/test_data/csv_data.csv")
        self.assertTrue("./tests/test_data/tsv_data.tsv")
        self.assertTrue("./tests/test_data/mat_data.mat")

    def test_input_data_list(self):
        self.hsic_lasso._input_data_list([[1, 1, 1],
                                          [2, 2, 2]
                                          ],
                                         [1, 2])
        X_in_row, X_in_col = self.hsic_lasso.X_in.shape
        Y_in_row, Y_in_col = self.hsic_lasso.Y_in.shape
        self.assertEqual(X_in_row, 3)
        self.assertEqual(X_in_col, 2)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 2)

        self.hsic_lasso._input_data_list([[1, 1, 1, 1, 1],
                                          [2, 2, 2, 2, 2],
                                          [3, 3, 3, 3, 3],
                                          [4, 4, 4, 4, 4]
                                          ],
                                         [1, 2, 3, 4])
        X_in_row, X_in_col = self.hsic_lasso.X_in.shape
        Y_in_row, Y_in_col = self.hsic_lasso.Y_in.shape
        self.assertEqual(X_in_row, 5)
        self.assertEqual(X_in_col, 4)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 4)

        with self.assertRaises(ValueError):
            self.hsic_lasso._input_data_list([[1, 1, 1, 1, 1],
                                              [2, 2, 2, 2, 2],
                                              [3, 3, 3, 3, 3],
                                              [4, 4, 4, 4, 4]
                                              ],
                                             [[1, 2, 3, 4],
                                              [1, 2, 3, 4]
                                              ])

    def test_input_data_ndarray(self):
        self.hsic_lasso._input_data_ndarray(np.array([[1, 1, 1],
                                                      [2, 2, 2]
                                                      ]),
                                            np.array([1, 2]))
        X_in_row, X_in_col = self.hsic_lasso.X_in.shape
        Y_in_row, Y_in_col = self.hsic_lasso.Y_in.shape
        self.assertEqual(X_in_row, 3)
        self.assertEqual(X_in_col, 2)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 2)

        self.hsic_lasso._input_data_ndarray(np.array([[1, 1, 1, 1, 1],
                                                      [2, 2, 2, 2, 2],
                                                      [3, 3, 3, 3, 3],
                                                      [4, 4, 4, 4, 4]
                                                      ]),
                                            np.array([1, 2, 3, 4]))
        X_in_row, X_in_col = self.hsic_lasso.X_in.shape
        Y_in_row, Y_in_col = self.hsic_lasso.Y_in.shape
        self.assertEqual(X_in_row, 5)
        self.assertEqual(X_in_col, 4)
        self.assertEqual(Y_in_row, 1)
        self.assertEqual(Y_in_col, 4)

        with self.assertRaises(ValueError):
            self.hsic_lasso._input_data_list(np.array([[1, 1, 1, 1, 1],
                                                       [2, 2, 2, 2, 2],
                                                       [3, 3, 3, 3, 3],
                                                       [4, 4, 4, 4, 4]
                                                       ]),
                                             np.array([[1, 2, 3, 4],
                                                       [1, 2, 3, 4]
                                                       ]))

    def test_check_shape(self):
        self.hsic_lasso._input_data_list([[1, 1, 1],
                                          [2, 2, 2]
                                          ],
                                         [1, 2])
        self.assertTrue(self.hsic_lasso._check_shape())

        self.hsic_lasso._input_data_list([[1, 1, 1, 1, 1],
                                          [2, 2, 2, 2, 2],
                                          [3, 3, 3, 3, 3],
                                          [4, 4, 4, 4, 4]
                                          ],
                                         [1, 2, 3, 4])
        self.assertTrue(self.hsic_lasso._check_shape())

        self.hsic_lasso._input_data_list([[1, 1, 1, 1],
                                          [2, 2, 2, 2],
                                          [3, 3, 3, 3],
                                          [4, 4, 4, 4]
                                          ],
                                         [1, 2, 3, 4])
        self.assertTrue(self.hsic_lasso._check_shape())

        self.hsic_lasso._input_data_list([[1, 1, 1, 1, 1],
                                          [2, 2, 2, 2, 2],
                                          [3, 3, 3, 3, 3]
                                          ],
                                         [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            self.hsic_lasso._check_shape()

    def test_input(self):
        self.assertTrue(self.hsic_lasso.input("./tests/test_data/csv_data.csv"))
        self.assertTrue(self.hsic_lasso.input("./tests/test_data/tsv_data.tsv"))
        self.assertTrue(self.hsic_lasso.input("./tests/test_data/matlab_data.mat"))


if __name__ == "__main__":
    unittest.main()
