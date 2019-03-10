#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import dict, range

from future import standard_library

import numpy as np
from joblib import Parallel, delayed

from .kernel_tools import kernel_delta_norm, kernel_gaussian

standard_library.install_aliases()


def compute_input_matrix(X_in, feature_idx, B, n, discarded, perms, x_kernel):

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    X = np.zeros((n * B * perms, 1), dtype=np.float32)

    st = 0
    ed = B ** 2
    index = np.arange(n)
    for p in range(perms):
        np.random.seed(p)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            # Normalization
            XX = X_in[index[i:j]]
            XX = XX.reshape((1, B))

            if x_kernel == 'Gauss':
                Kx = kernel_gaussian(XX[0, None], XX[0, None], 1.0)
            elif x_kernel == 'Delta':
                Kx = kernel_delta_norm(XX[0, None], XX[0, None])

            tmp = np.dot(np.dot(H, Kx), H)

            # Normalize HSIC tr(tmp*tmp) = 1
            tmp = tmp / (np.linalg.norm(tmp, 'fro') + 10e-10)
            X[st:ed, 0] = tmp.flatten()
            st += B ** 2
            ed += B ** 2

    return (feature_idx, X.flatten())


def hsic_lasso(X, Y, y_kernel, x_kernel='Gauss', n_jobs=-1, discarded=0, B=0, M=1):
    """
    Input:
        X      input_data
        Y      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size d x (n * B (or n) * M)
        X_ty      vector of size d x 1
    """
    d, n = X.shape
    dy = Y.shape[0]

    L = np.zeros((n * B * M, dy))
    for i in range(dy):
        L[:,i] = compute_kernel(Y[i,:], y_kernel, B, M, discarded)

    # Preparing design matrix for HSIC Lars
    result = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        X[k, :], x_kernel, k, B, M, n, discarded) for k in range(d)])

    # non-parallel version for debugging purposes
    # result = []
    # for k in range(d):
    #     X = parallel_compute_kernel(X[k, :], x_kernel, k, B, M, n, discarded)
    #     result.append(X)

    result = dict(result)

    K = np.array([result[k] for k in range(d)]).T
    KtL = np.dot(K.T, L)

    return K, KtL

def compute_kernel(x, kernel, B = 0, M = 1, discarded = 0):

    n = x.shape[0]

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    # Normalize data
    if kernel == "Gauss":
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = B ** 2
    index = np.arange(n)
    for p in range(M):
        np.random.seed(p)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            if kernel == 'Gauss':
                try:
                    Kx = kernel_gaussian(x[index[i:j]], x[index[i:j]], 1.0)
                except:
                    import pdb; pdb.set_trace()
            elif kernel == 'Delta':
                Kx = kernel_delta_norm(x[index[i:j]], x[index[i:j]])

            tmp = np.dot(np.dot(H, Kx), H)

            # Normalize HSIC tr(tmp*tmp) = 1
            tmp = tmp / (np.linalg.norm(tmp, 'fro') + 10e-10)
            K[st:ed] = tmp.flatten()
            st += B ** 2
            ed += B ** 2

    return K

def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded):

    return (feature_idx, compute_kernel(x, kernel, B, M, discarded))