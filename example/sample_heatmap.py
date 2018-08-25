#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from pyHSICLasso import HSICLasso

standard_library.install_aliases()


def main():
    hsic_lasso = HSICLasso()
    hsic_lasso.input("../tests/test_data/matlab_data.mat")

    #max_neighbors=0 means that we only use the HSIC Lasso features to plot heatmap
    hsic_lasso.regression(5,max_neighbors=0)

    #Compute linkage
    hsic_lasso.linkage()

    #Run Hierarchical clustering
    # Features are clustered by using HSIC scores
    # Samples are clusterd by using Euclid distance
    hsic_lasso.plot_heatmap()

if __name__ == "__main__":
    main()
