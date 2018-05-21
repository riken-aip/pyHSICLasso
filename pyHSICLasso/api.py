#!usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

import numpy as np
from future import standard_library
from six import string_types

from .hsic_lasso import hsic_lasso
from .input_data import input_csv_file, input_matlab_file, input_tsv_file
from .nlars import nlars
from .plot_figure import plot_figure

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

    def input(self, *args):
        self._check_args(args)
        if isinstance(args[0], string_types):
            self._input_data_file(args[0])
        elif isinstance(args[0], list):
            self._input_data_list(args[0], args[1])
        elif isinstance(args[0], np.ndarray):
            self._input_data_ndarray(args[0], args[1])
        else:
            pass
        if self.X_in is None or self.Y_in is None:
            raise ValueError("Check your input data")
        self._check_shape()
        return True

    def regression(self, num_feat=5):
        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        self.X, self.X_ty = hsic_lasso(self.X_in, self.Y_in, "Gauss")
        self.path, self.beta, self.A, self.lam, \
        self.A_neighbors, self.A_neighbors_score = nlars(self.X,
                                                         self.X_ty, num_feat)
        return True

    def classification(self, num_feat=5):
        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        self.Y_in = (np.sign(self.Y_in) + 1) / 2 + 1
        self.X, self.X_ty = hsic_lasso(self.X_in, self.Y_in, "Delta")
        self.path, self.beta, self.A, self.lam, \
        self.A_neighbors, self.A_neighbors_score = nlars(self.X,
                                                         self.X_ty, num_feat)
        return True

    def dump(self):

        #To normalize the feature importance
        maxval = self.path[self.A[0],-1:][0]
        print("============================================== HSICLasso : Result ==================================================")
        print("| Order | Feature      | Score | Top-5 Related Feature (Relatedness Score)                                          |")
        for i in range(len(self.A)):
            print("| {:<5} | {:<12} | {:.3f} | {:<12} ({:.3f}), {:<12} ({:.3f}),"
                  " {:<12} ({:.3f}), {:<12} ({:.3f}), {:<12} ({:.3f})|".format(i+1,self.featname[self.A[i]],
                                                                            self.path[self.A[i],-1:][0]/maxval,
                                                                            self.featname[self.A_neighbors[i][1]],self.A_neighbors_score[i][1],
                                                                            self.featname[self.A_neighbors[i][2]],self.A_neighbors_score[i][2],
                                                                            self.featname[self.A_neighbors[i][3]],self.A_neighbors_score[i][3],
                                                                            self.featname[self.A_neighbors[i][4]],self.A_neighbors_score[i][4],
                                                                            self.featname[self.A_neighbors[i][5]],self.A_neighbors_score[i][5]))

        #print("===== HSICLasso : Path ======")
        #for i in range(len(self.A)):
        #    print(self.path[self.A[i], 1:])
        #return True

    def plot(self):
        if self.path is None or self.beta is None or self.A is None:
            raise UnboundLocalError("Input your data")
        plot_figure(self.path, self.beta, self.A)
        return True

    def get_index(self):
        return self.A

    def get_index_neighbors(self,feat_index=0,num_neighbors=5):
        if feat_index > len(self.A) -1:
            raise ValueError("Index does not exist")

        num_neighbors = min(num_neighbors,10)

        return self.A_neighbors[feat_index][1:(num_neighbors+1)]

    def get_index_neighbors_score(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            raise ValueError("Index does not exist")

        num_neighbors = min(num_neighbors, 10)

        return self.A_neighbors_score[feat_index][1:(num_neighbors + 1)]

    def save_score(self,filename='aggregated_score.csv'):
        maxval = self.path[self.A[0], -1:][0]
        fout = open(filename,'w')
        featscore = {}
        featcorrcoeff = {}
        for i in range(len(self.A)):
            HSIC_XY = (self.path[self.A[i], -1:][0] / maxval)

            if self.featname[self.A[i]] not in featscore:
                featscore[self.featname[self.A[i]]] = HSIC_XY

                corrcoeff = np.corrcoef(self.X_in[self.A[i]],self.Y_in)[0][1]

                featcorrcoeff[self.featname[self.A[i]]] = corrcoeff

            else:
                featscore[self.featname[self.A[i]]] += HSIC_XY

            for j in range(1, 11):
                HSIC_XX = self.A_neighbors_score[i][j]
                if self.featname[self.A_neighbors[i][j]] not in featscore:
                    featscore[self.featname[self.A_neighbors[i][j]]] = HSIC_XY * HSIC_XX

                    corrcoeff = np.corrcoef(self.X_in[self.A_neighbors[i][j]], self.Y_in)[0][1]

                    featcorrcoeff[self.featname[self.A_neighbors[i][j]]] = corrcoeff
                else:
                    featscore[self.featname[self.A_neighbors[i][j]]] += HSIC_XY * HSIC_XX

        # Sorting decending order
        featscore_sorted = sorted(featscore.items(), key=lambda x: x[1], reverse=True)

        # Add Pearson correlation for comparison
        fout.write('Feature,Score,Pearson Corr\n')
        for (key, val) in featscore_sorted:
            fout.write(key + ',' + str(val) + ',' + str(featcorrcoeff[key]) + '\n')

        fout.close()

    def save_param(self,filename='param.csv'):
        # Save parameters
        maxval = self.path[self.A[0], -1:][0]

        fout = open(filename, 'w')
        sstr = 'Feature,Score,'
        for j in range(1, 11):
            sstr = sstr + 'Neighbor %d, Neighbor %d score,' % (j, j)

        sstr = sstr + '\n'
        fout.write(sstr)
        for i in range(len(self.A)):
            tmp = []
            tmp.append(self.featname[self.A[i]])
            tmp.append(str(self.path[self.A[i], -1:][0] / maxval))
            for j in range(1, 11):
                tmp.append(str(self.featname[self.A_neighbors[i][j]]))
                tmp.append(str(self.A_neighbors_score[i][j]))

            sstr = ','.join(tmp) + '\n'
            fout.write(sstr)

        fout.close()

    # ========================================

    def _check_args(self, args):
        if len(args) == 0 or len(args) >= 3:
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
        return True

    def _input_data_file(self, file_name):
        ext = file_name[-4:]
        if ext == ".csv":
            self.X_in, self.Y_in, self.featname = input_csv_file(file_name)
        elif ext == ".tsv":
            self.X_in, self.Y_in, self.featname = input_tsv_file(file_name)
        elif ext == ".mat":
            self.X_in, self.Y_in, self.featname = input_matlab_file(file_name)
        return True

    def _input_data_list(self, X_in, Y_in):
        if isinstance(Y_in[0], list):
            raise ValueError("Check your input data")
        self.X_in = np.array(X_in).T
        self.Y_in = np.array(Y_in).reshape(1, len(Y_in))
        return True

    def _input_data_ndarray(self, X_in, Y_in):
        if len(Y_in.shape) == 2:
            raise ValueError("Check your input data")
        self.X_in = X_in.T
        self.Y_in = Y_in.reshape(1, len(Y_in))
        return True

    def _check_shape(self):
        x_row_len, x_col_len = self.X_in.shape
        y_row_len, y_col_len = self.Y_in.shape
        if y_row_len != 1:
            raise ValueError("Check your input data")
        if x_col_len != y_col_len:
            raise ValueError("Check your input data")
        return True
