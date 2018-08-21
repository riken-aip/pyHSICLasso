#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

import numpy as np

standard_library.install_aliases()


def kernel_delta_norm(X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))

    u_list = np.unique(X_in_1)
    for ind in u_list:
        ind_1 = (X_in_1 == ind)
        ind_2 = (X_in_2 == ind)
        idx = (ind_1 & ind_2.T)
        c_1 = np.sqrt(np.count_nonzero(ind_1))
        c_2 = np.sqrt(np.count_nonzero(ind_2))
        K[idx] = 1 / c_1 / c_2
    return K


def kernel_delta(X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    u_list = np.unique(X_in_1)
    for ind in u_list:
        ind_1 = (X_in_1 == ind)
        ind_2 = (X_in_2 == ind)
        idx = (ind_1 & ind_2.T)
        K[idx] = 1
    return K


def kernel_gaussian(X_in_1, X_in_2, sigma):
    X_in_12 = np.sum(np.power(X_in_1, 2), 0)
    X_in_12 = np.expand_dims(X_in_12, 0)
    X_in_22 = np.sum(np.power(X_in_2, 2), 0)
    X_in_22 = np.expand_dims(X_in_22, 0)
    dist_2 = (X_in_12 + X_in_22.T) - 2 * np.dot(X_in_1.T, X_in_2)
    K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
    return K
