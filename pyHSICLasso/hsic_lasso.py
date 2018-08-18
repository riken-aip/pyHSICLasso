#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

import numpy as np
from joblib import Parallel, delayed
from future import standard_library

from .kernel_tools import kernel_delta_norm, kernel_gaussian

standard_library.install_aliases()

def process(X_in,ii,B,n,discarded,perms,x_kernel):
    st = 0
    ed = B ** 2

    H = np.eye(B) - 1 / B * np.ones(B)

    X = np.zeros((n*B*perms,1))
    index = list(range(0, n))
    for p in range(perms):
        np.random.seed(p)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            # Normalization
            if x_kernel == 'Gauss':
                XX = X_in[index[i:j]] / (X_in[index[i:j]].std() + 10e-20) * np.sqrt(float(B - 1) / B)
            else:
                XX = X_in[index[i:j]]

            XX = XX.reshape((1, B))

            if x_kernel == 'Gauss':
                Kx = kernel_gaussian(XX[0, None], XX[0, None], 1.0)
            elif x_kernel == 'Delta':
                Kx = kernel_delta_norm(XX[0, None], XX[0, None])

            tmp = np.dot(np.dot(H, Kx), H)

            # Normalize HSIC tr(tmp*tmp) = 1
            tmp = tmp / np.linalg.norm(tmp, 'fro')
            X[st:ed, 0] = tmp.flatten()
            st += B ** 2
            ed += B ** 2

    return [{'id': ii, 'val': X.flatten()}]


def hsic_lasso(X_in, Y_in,y_kernel, x_kernel = 'Gauss',discarded=0,B=0,perms=1):
    """
    Input:
        X_in      input_data
        Y_in      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size d x (n * B (or n) * perms)
        X_ty      vector of size d x 1
    """
    d, n = X_in.shape
    dy = Y_in.shape[0]

    # Centering matrix
    H = np.eye(B) - 1 / B * np.ones(B)
    lf = np.zeros((n * B * perms, 1))
    index = list(range(0,n))
    st = 0
    ed = B**2
    for p in range(perms):
        np.random.seed(p)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            if y_kernel == "Delta":
                if dy > 1:
                    raise ValueError("Delta kernel only supports 1 dimensional class labels.")

                L = kernel_delta_norm(Y_in[:,index[i:j]], Y_in[:,index[i:j]])
            elif y_kernel == "Gauss":
                YY = Y_in[:,index[i:j]] / (Y_in[:,index[i:j]].std(1)[:, None] + 10e-20) * np.sqrt(float(B - 1) / B)
                L = kernel_gaussian(YY, YY, np.sqrt(dy))

            L = np.dot(H, np.dot(L, H))

            #Normalize HSIC tr(L*L) = 1
            L = L / np.linalg.norm(L, 'fro')

            lf[st:ed,0] = L.flatten()
            st += B**2
            ed += B**2

    # Preparing design matrix for HSIC Lars
    result = Parallel(n_jobs=-1)([delayed(process)(X_in[ii,:],ii,B,n,discarded,perms,x_kernel) for ii in range(d)])

    X = np.zeros((n * B * perms, d))
    for tmp in result:
        id = tmp[0]['id']
        X[:,id] = tmp[0]['val']

    X_ty = np.dot(X.T,lf)


    return X, X_ty
