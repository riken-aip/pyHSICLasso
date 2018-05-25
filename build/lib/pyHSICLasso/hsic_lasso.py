#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

import numpy as np
from future import standard_library

from .kernel_tools import kernel_delta_norm, kernel_gaussian

standard_library.install_aliases()


def hsic_lasso(X_in, Y_in, y_kernel):
    """
    Input:
        X_in      input_data
        Y_in      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size D x D
        X_ty      vector of size D x 1
    """
    d, n = X_in.shape

    # Centering matrix
    H = np.eye(n) - 1 / n * np.ones(n)

    # Normalization
    XX = X_in / (X_in.std(1)[:, None]) * np.sqrt(n - 1 / n)

    if y_kernel == "Delta":
        L = kernel_delta_norm(Y_in, Y_in)
    elif y_kernel == "Gauss":
        YY = Y_in / (Y_in.std(1)[:, None] + 10e-20) * np.sqrt(n - 1 / n)
        L = kernel_gaussian(YY, YY, 1.0)

    L = np.dot(H, np.dot(L, H))

    #Normalize HSIC tr(L*L) = 1
    L = L / np.linalg.norm(L, 'fro')

    # Preparing design matrix for HSIC Lars
    X = np.zeros((n * n, d))
    X_ty = np.zeros((d, 1))
    for ii in range(d):
        Kx = kernel_gaussian(XX[ii, None], XX[ii, None], 1.0)
        tmp = np.dot(np.dot(H, Kx), H)

        #Normalize HSIC tr(tmp*tmp) = 1
        tmp = tmp / np.linalg.norm(tmp, 'fro')
        X[:, ii] = tmp.flatten()
        X_ty[ii] = (tmp * L).sum()

    return X, X_ty
