#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
from builtins import int, open, range, str

from future import standard_library

import numpy as np
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import linkage
from six import string_types

from .hsic_lasso import hsic_lasso, compute_kernel
from .input_data import input_csv_file, input_matlab_file, input_tsv_file
from .nlars import nlars
from .plot_figure import plot_dendrogram, plot_heatmap, plot_path

standard_library.install_aliases()


class HSICLasso(object):
    def __init__(self):
        self.input_file = None
        self.X_in = None
        self.Y_in = None
        self.X = None
        self.Xty = None
        self.path = None
        self.beta = None
        self.A = None
        self.A_neighbors = None
        self.A_neighbors_score = None
        self.lam = None
        self.featname = None
        self.linkage_dist = None
        self.hclust_featname = None
        self.hclust_featnameindex = None
        self.max_neighbors = 10

    def input(self, *args, **_3to2kwargs):
        if 'output_list' in _3to2kwargs: output_list = _3to2kwargs['output_list']; del _3to2kwargs['output_list']
        else: output_list = ['class']

        self._check_args(args)
        if isinstance(args[0], string_types):
            self._input_data_file(args[0], output_list)
        elif isinstance(args[0], np.ndarray):
            if 'featname' in _3to2kwargs:
                featname = _3to2kwargs['featname']; del _3to2kwargs['featname']
            else: featname = ['%d' % x for x in range(1, args[0].shape[1] + 1)]

            if len(args) == 2:
                self._input_data_ndarray(args[0], args[1], featname)
            if len(args) == 3:
                self._input_data_ndarray(args[0], args[1], args[2])
        else:
            pass
        if self.X_in is None or self.Y_in is None:
            raise ValueError("Check your input data")
        self._check_shape()
        return True

    def regression(self, num_feat=5, B=20, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1, covars = np.array([]),covars_kernel="Gaussian"):
        self._run_hsic_lasso(num_feat=num_feat,
                             y_kernel="Gaussian",
                             B=B, M=M,
                             discrete_x=discrete_x,
                             max_neighbors=max_neighbors,
                             n_jobs=n_jobs,
                             covars=covars,
                             covars_kernel=covars_kernel)

        return True

    def classification(self, num_feat=5, B=20, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1, covars = np.array([]),covars_kernel="Gaussian"):
        self._run_hsic_lasso(num_feat=num_feat,
                             y_kernel="Delta",
                             B=B, M=M,
                             discrete_x=discrete_x,
                             max_neighbors=max_neighbors,
                             n_jobs=n_jobs, 
                             covars=covars,
                             covars_kernel=covars_kernel)

        return True

    def _run_hsic_lasso(self, y_kernel, num_feat, B, M, discrete_x, max_neighbors, n_jobs, covars, covars_kernel):

        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        self.max_neighbors = max_neighbors
        n = self.X_in.shape[1]
        B = B if B else n
        x_kernel = "Delta" if discrete_x else "Gaussian"
        numblocks = n / B
        discarded = n % B

        print('Block HSIC Lasso B = {}.'.format(B))

        if discarded:
            msg = "B {} must be an exact divisor of the number of samples {}. Number \
of blocks {} will be approximated to {}.".format(B, n, numblocks, int(numblocks))
            warnings.warn(msg, RuntimeWarning)
            numblocks = int(numblocks)

        # Number of permutations of the block HSIC
        M = 1 + bool(numblocks - 1) * (M - 1)
        print('M set to {}.'.format(M))
        print('Using {} kernel for the features, {} kernel for the outcomes{}.'.format(
            x_kernel, y_kernel, ' and Gaussian kernel for the covariates' if covars.size else ''))

        X,Xty,Ky = hsic_lasso(self.X_in, self.Y_in, y_kernel, x_kernel,
                              n_jobs=n_jobs, discarded=discarded, B=B, M=M)

        # np.concatenate(self.X, axis = 0) * np.sqrt(1/(numblocks * M))
        self.X = X * np.sqrt(1 / (numblocks * M))
        self.Xty = Xty * 1 / (numblocks * M)

        if covars.size:
            if self.X_in.shape[1] != covars.shape[0]:
                raise UnboundLocalError("The number of rows in the covars matrix should be " + str(self.X_in.shape[1]))

            if covars_kernel == "Gaussian":
                Kc = compute_kernel(covars.transpose(), 'Gaussian', B, M, discarded)
            else:
                Kc = compute_kernel(covars.transpose(), 'Delta', B, M, discarded)
            Kc = np.reshape(Kc,(n * B * M,1))

            Ky = Ky * np.sqrt(1 / (numblocks * M))
            Kc = Kc * np.sqrt(1 / (numblocks * M))

            betas = np.dot(Ky.transpose(),Kc) / np.trace(np.dot(Kc.T, Kc))
            #print(betas)
            self.Xty = self.Xty - betas*np.dot(self.X.transpose(),Kc)

        self.path, self.beta, self.A, self.lam, self.A_neighbors, \
            self.A_neighbors_score = nlars(
                self.X, self.Xty, num_feat, self.max_neighbors)

        return True

    # For kernel Hierarchical Clustering
    def linkage(self, method="ward"):
        if self.A is None:
            raise UnboundLocalError("Run regression/classification first")
        # selected feature name
        featname_index = []
        featname_selected = []
        for i in range(len(self.A) - 1):
            for index in self.A_neighbors[i]:
                if index not in featname_index:
                    featname_index.append(index)
                    featname_selected.append(self.featname[index])
        self.hclust_featname = featname_selected
        self.hclust_featnameindex = featname_index
        sim = np.dot(self.X[:, featname_index].transpose(),
                     self.X[:, featname_index])
        dist = 1 - sim
        dist = np.maximum(0, dist - np.diag(np.diag(dist)))
        dist_sym = (dist + dist.transpose()) / 2.0
        self.linkage_dist = linkage(distance.squareform(dist_sym), method)

        return True

    def dump(self):
        maxval = self.beta[self.A[0]][0]
        print("============================================== HSICLasso : Result ==================================================")
        print("| Order | Feature      | Score | Top-5 Related Feature (Relatedness Score)                                          |")
        for i in range(len(self.A)):
            print("| {:<5} | {:<12} | {:.3f} | {:<12} ({:.3f}), {:<12} ({:.3f}),"
                  " {:<12} ({:.3f}), {:<12} ({:.3f}), {:<12} ({:.3f})|".format(i + 1, self.featname[self.A[i]],
                                                                               self.beta[self.A[i]
                                                                                         ][0] / maxval,
                                                                               self.featname[self.A_neighbors[i][1]
                                                                                             ], self.A_neighbors_score[i][1],
                                                                               self.featname[self.A_neighbors[i][2]
                                                                                             ], self.A_neighbors_score[i][2],
                                                                               self.featname[self.A_neighbors[i][3]
                                                                                             ], self.A_neighbors_score[i][3],
                                                                               self.featname[self.A_neighbors[i][4]
                                                                                             ], self.A_neighbors_score[i][4],
                                                                               self.featname[self.A_neighbors[i][5]], self.A_neighbors_score[i][5]))

        #print("===== HSICLasso : Path ======")
        # for i in range(len(self.A)):
        #    print(self.path[self.A[i], 1:])
        # return True

    def plot_heatmap(self, filepath = 'heatmap.png'):
        if self.linkage_dist is None or self.hclust_featname is None or self.hclust_featnameindex is None:
            raise UnboundLocalError("Input your data")
        plot_heatmap(self.X_in[self.hclust_featnameindex, :],
                     self.linkage_dist, self.hclust_featname,
                     filepath)
        return True

    def plot_dendrogram(self, filepath = 'dendrogram.png'):
        if self.linkage_dist is None or self.hclust_featname is None:
            raise UnboundLocalError("Input your data")
        plot_dendrogram(self.linkage_dist, self.hclust_featname, filepath)
        return True

    def plot_path(self, filepath = 'path.png'):
        if self.path is None or self.beta is None or self.A is None:
            raise UnboundLocalError("Input your data")
        plot_path(self.path, self.beta, self.A, filepath)
        return True

    def get_features(self):
        index = self.get_index()

        return [self.featname[i] for i in index]

    def get_features_neighbors(self, feat_index=0, num_neighbors=5):
        index = self.get_index_neighbors(
            feat_index=feat_index, num_neighbors=num_neighbors)

        return [self.featname[i] for i in index]

    def get_index(self):
        return self.A

    def get_index_score(self):
        return self.beta[self.A, -1]

    def get_index_neighbors(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            raise IndexError("Index does not exist")

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors[feat_index][1:(num_neighbors + 1)]

    def get_index_neighbors_score(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            raise IndexError("Index does not exist")

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors_score[feat_index][1:(num_neighbors + 1)]

    def save_HSICmatrix(self, filename='HSICmatrix.csv'):
        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")

        self.X, self.X_ty = hsic_lasso(self.X_in, self.Y_in, "Gaussian")

        K = np.dot(self.X.transpose(), self.X)

        np.savetxt(filename, K, delimiter=',', fmt='%.7f')

        return True

    def save_score(self, filename='aggregated_score.csv'):
        maxval = self.beta[self.A[0]][0]

        #print(maxval + ' ' + maxval_)
        fout = open(filename, 'w')
        featscore = {}
        featcorrcoeff = {}
        for i in range(len(self.A)):
            HSIC_XY = (self.beta[self.A[i]][0] / maxval)

            if self.featname[self.A[i]] not in featscore:
                featscore[self.featname[self.A[i]]] = HSIC_XY

                corrcoeff = np.corrcoef(self.X_in[self.A[i]], self.Y_in)[0][1]

                featcorrcoeff[self.featname[self.A[i]]] = corrcoeff

            else:
                featscore[self.featname[self.A[i]]] += HSIC_XY

            for j in range(1, self.max_neighbors + 1):
                HSIC_XX = self.A_neighbors_score[i][j]
                if self.featname[self.A_neighbors[i][j]] not in featscore:
                    featscore[self.featname[self.A_neighbors[i][j]]
                              ] = HSIC_XY * HSIC_XX

                    corrcoeff = np.corrcoef(
                        self.X_in[self.A_neighbors[i][j]], self.Y_in)[0][1]

                    featcorrcoeff[self.featname[self.A_neighbors[i]
                                                [j]]] = corrcoeff
                else:
                    featscore[self.featname[self.A_neighbors[i][j]]
                              ] += HSIC_XY * HSIC_XX

        # Sorting decending order
        featscore_sorted = sorted(
            featscore.items(), key=lambda x: x[1], reverse=True)

        # Add Pearson correlation for comparison
        fout.write('Feature,Score,Pearson Corr\n')
        for (key, val) in featscore_sorted:
            fout.write(key + ',' + str(val) + ',' +
                       str(featcorrcoeff[key]) + '\n')

        fout.close()

    def save_param(self, filename='param.csv'):
        # Save parameters
        maxval = self.beta[self.A[0]][0]

        fout = open(filename, 'w')
        sstr = 'Feature,Score,'
        for j in range(1, self.max_neighbors + 1):
            sstr = sstr + 'Neighbor %d, Neighbor %d score,' % (j, j)

        sstr = sstr + '\n'
        fout.write(sstr)
        for i in range(len(self.A)):
            tmp = []
            tmp.append(self.featname[self.A[i]])
            tmp.append(str(self.beta[self.A[i]][0] / maxval))
            for j in range(1, self.max_neighbors + 1):
                tmp.append(str(self.featname[self.A_neighbors[i][j]]))
                tmp.append(str(self.A_neighbors_score[i][j]))

            sstr = ','.join(tmp) + '\n'
            fout.write(sstr)

        fout.close()

    # ========================================

    def _check_args(self, args):
        if len(args) == 0 or len(args) >= 4:
            raise SyntaxError("Input as input_data(file_name) or \
                input_data(X_in, Y_in)")
        elif len(args) == 1:
            if isinstance(args[0], string_types):
                if len(args[0]) <= 4:
                    raise ValueError("Check your file name")
                else:
                    ext = args[0][-4:]
                    if ext == ".csv" or ext == ".tsv" or ext == ".mat":
                        pass
                    else:
                        raise TypeError("Input file is only .csv, .tsv .mat")
            else:
                raise TypeError("File name is only str")
        elif len(args) == 2:
            if isinstance(args[0], string_types):
                raise TypeError("Check arg type")
            elif isinstance(args[0], list):
                if isinstance(args[1], list):
                    pass
                else:
                    raise TypeError("Check arg type")
            elif isinstance(args[0], np.ndarray):
                if isinstance(args[1], np.ndarray):
                    pass
                else:
                    raise TypeError("Check arg type")
            else:
                raise TypeError("Check arg type")
        elif len(args) == 3:
            if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray) and isinstance(args[2], list):
                pass
            else:
                raise TypeError("Check arg type")

        return True

    def _input_data_file(self, file_name, output_list):
        ext = file_name[-4:]
        if ext == ".csv":
            self.X_in, self.Y_in, self.featname = input_csv_file(
                file_name, output_list=output_list)
        elif ext == ".tsv":
            self.X_in, self.Y_in, self.featname = input_tsv_file(
                file_name, output_list=output_list)
        elif ext == ".mat":
            self.X_in, self.Y_in, self.featname = input_matlab_file(file_name)
        return True

    def _input_data_list(self, X_in, Y_in):
        if isinstance(Y_in[0], list):
            raise ValueError("Check your input data")
        self.X_in = np.array(X_in).T
        self.Y_in = np.array(Y_in).reshape(1, len(Y_in))
        return True

    def _input_data_ndarray(self, X_in, Y_in, featname = None):
        if len(Y_in.shape) == 2:
            raise ValueError("Check your input data")
        self.X_in = X_in.T
        self.Y_in = Y_in.reshape(1, len(Y_in))
        self.featname = featname
        return True

    def _check_shape(self):
        _, x_col_len = self.X_in.shape
        y_row_len, y_col_len = self.Y_in.shape
        # if y_row_len != 1:
        #    raise ValueError("Check your input data")
        if x_col_len != y_col_len:
            raise ValueError(
                "The number of samples in input and output should be same")
        return True

    def _permute_data(self, seed=None):
        np.random.seed(seed)
        n = self.X_in.shape[1]

        perm = np.random.permutation(n)
        self.X_in = self.X_in[:, perm]
        self.Y_in = self.Y_in[:, perm]
