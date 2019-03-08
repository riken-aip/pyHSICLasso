#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

from future import standard_library

import numpy as np
from scipy.sparse import lil_matrix

standard_library.install_aliases()


def nlars(X, X_ty, num_feat, max_neighbors):
    """
    We used the a Python implementation of the Nonnegative LARS solver
    written in MATLAB at http://orbit.dtu.dk/files/5618980/imm5523.zip

    Solves the problem argmin_beta 1/2||y-X*beta||_2^2  s.t. beta>=0.
    The problem is solved using a modification of the Least Angle Regression
    and Selection algorithm.
    As such the entire regularization path for the LASSO problem
    min 1/2||y-X*beta||_2^2 + lambda|beta|_1  s.t. beta>=0
    for all values of lambda is given in path.

    Input:
        X            matrix of size D x D
        X_ty         vector of size D x 1
        num_feat     the number of features you want to extract
    Output:
        path         the entire solution path
        beta         D x 1 solution vector
        A            selected features
        A_neighbors  related features of the selected features in A
        lam(lambda)  regularization value at beginning of step corresponds
                     to value of negative gradient
    """
    n, d = X.shape

    A = []
    A_neighbors = []
    A_neighbors_score = []
    beta = np.zeros((d, 1), dtype=np.float32)
    path = lil_matrix((d, 4 * d))
    lam = np.zeros((1, 4 * d))

    I = list(range(d))

    XtXbeta = np.dot(X.transpose(), np.dot(X, beta))
    c = X_ty - XtXbeta
    j = c.argmax()
    C = c[j]
    A.append(I[j])
    I.remove(I[j])

    if len(C) == 0:
        lam[0] = 0
    else:
        lam[0, 0] = C[0]

    k = 0
    while sum(c[A]) / len(A) >= 1e-9 and len(A) < num_feat + 1:
        s = np.ones((len(A), 1), dtype=np.float32)

        try:
            w = np.linalg.solve(np.dot(X[:, A].transpose(), X[:, A]), s)
        except np.linalg.linalg.LinAlgError:
            # matrix is singular
            X_noisy = X[:, A] + np.random.normal(0, 10e-10, X[:, A].shape)
            w = np.linalg.solve(np.dot(X_noisy.transpose(), X_noisy), s) 

        XtXw = np.dot(X.transpose(), np.dot(X[:, A], w))

        gamma1 = (C - c[I]) / (XtXw[A[0]] - XtXw[I])
        gamma2 = -beta[A] / (w)
        gamma3 = np.zeros((1, 1))
        gamma3[0] = c[A[0]] / (XtXw[A[0]])
        gamma = np.concatenate((np.concatenate((gamma1, gamma2)), gamma3))

        gamma[gamma <= 1e-9] = np.inf
        t = gamma.argmin()
        mu = min(gamma)

        beta[A] = beta[A] + mu * w

        if t >= len(gamma1) and t < (len(gamma1) + len(gamma2)):
            lasso_cond = 1
            j = t - len(gamma1)
            I.append(A[j])
            A.remove(A[j])
        else:
            lasso_cond = 0

        XtXbeta = np.dot(X.transpose(), np.dot(X, beta))
        c = X_ty - XtXbeta
        j = np.argmax(c[I])
        C = max(c[I])

        k += 1
        path[:, k] = beta

        if len(C) == 0:
            lam[k] = 0
        else:
            lam[0, k] = C[0]
        if lasso_cond == 0:
            A.append(I[j])
            I.remove(I[j])

    # We run numfeat + 1 iteration to update beta and path information
    # Then, we return only numfeat features
    if len(A) > num_feat:
        A.pop()

    # Sort A with respect to beta
    s = beta[A]
    sort_index = sorted(range(len(s)), key=lambda k: s[k], reverse=True)

    A_sorted = [A[i] for i in sort_index]

    # Find nighbors of selected features
    XtXA = np.dot(X.transpose(), X[:, A_sorted])

    # Search up to 10 nighbors
    num_neighbors = max_neighbors + 1
    for i in range(0, len(A_sorted)):
        tmp = XtXA[:, i]
        sort_index = sorted(
            range(len(tmp)), key=lambda k: tmp[k], reverse=True)
        A_neighbors.append(sort_index[0:num_neighbors])
        A_neighbors_score.append(tmp[sort_index[0:num_neighbors]])

    path_final = path[:, 0:(k + 1)].toarray()
    lam_final = lam[0:(k + 1)]

    return path_final, beta, A_sorted, lam_final, A_neighbors, A_neighbors_score
