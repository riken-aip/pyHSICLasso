#!/usr/bin.env python
# coding: utf-8

from api import HSICLasso

def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input("../tests/test_data/matlab_data.mat")
    hsic_lasso.regression(5)
    hsic_lasso.dump()
    hsic_lasso.plot()


if __name__ == "__main__":
    main()
