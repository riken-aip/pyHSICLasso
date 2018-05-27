#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from api import HSICLasso

standard_library.install_aliases()


def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input("../tests/test_data/matlab_data.mat")
    hsic_lasso.regression(5)
    hsic_lasso.dump()
    hsic_lasso.plot()


if __name__ == "__main__":
    main()
