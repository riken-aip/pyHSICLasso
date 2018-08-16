#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

import numpy as np
from future import standard_library

from .kernel_tools import kernel_delta_norm, kernel_gaussian

standard_library.install_aliases()


def hsic_lasso(X_in, Y_in, y_kernel, x_kernel = 'Gauss'):
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
    if x_kernel == 'Gauss':
        XX = X_in / (X_in.std(1)[:, None] + 10e-20) * np.sqrt(float(n - 1) / n)
    else:
        XX = X_in

    dy = Y_in.shape[0]

    if y_kernel == "Delta":
        if dy > 1:
            raise ValueError("Delta kernel only supports 1 dimensional class labels.")

        L = kernel_delta_norm(Y_in, Y_in)
    elif y_kernel == "Gauss":
        YY = Y_in / (Y_in.std(1)[:, None] + 10e-20) * np.sqrt(float(n - 1) / n)
        L = kernel_gaussian(YY, YY, np.sqrt(dy))

    L = np.dot(H, np.dot(L, H))

    #Normalize HSIC tr(L*L) = 1
    L = L / np.linalg.norm(L, 'fro')

    # Preparing design matrix for HSIC Lars
    X = np.zeros((n * n, d))
    X_ty = np.zeros((d, 1))
    for ii in range(d):
        if x_kernel == 'Gauss':
            Kx = kernel_gaussian(XX[ii, None], XX[ii, None], 1.0)
        elif x_kernel == 'Delta':
            Kx = kernel_delta_norm(XX[ii, None], XX[ii, None])

        tmp = np.dot(np.dot(H, Kx), H)

        #Normalize HSIC tr(tmp*tmp) = 1
        tmp = tmp / np.linalg.norm(tmp, 'fro')
        X[:, ii] = tmp.flatten()
        X_ty[ii] = (tmp * L).sum()

    return X, X_ty
