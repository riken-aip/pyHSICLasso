#!usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
from future import standard_library
from scipy import io as spio

standard_library.install_aliases()


def input_csv_file(file_name):
    df = pd.read_csv(file_name, sep=",")

    #Store the column name (Feature name)
    featname = df.columns[1:].tolist()

    X_in = df.ix[:, 1:].as_matrix().T
    Y_in = df.ix[:, 0].as_matrix().reshape(1, len(df.index))
    return X_in, Y_in, featname


def input_tsv_file(file_name):
    df = pd.read_csv(file_name, sep="\t")

    # Store the column name (Feature name)
    featname = df.columns[1:].tolist()

    X_in = df.ix[:, 1:].as_matrix().T
    Y_in = df.ix[:, 0].as_matrix().reshape(1, len(df.index))
    return X_in, Y_in, featname


def input_matlab_file(file_name):
    data = spio.loadmat(file_name)



    if "X" in data.keys() and "Y" in data.keys():
        X_in = data["X"]
        Y_in = data["Y"]
    elif "X_in" in data.keys() and "Y_in" in data.keys():
        X_in = data["X_in"]
        Y_in = data["Y_in"]
    elif "x" in data.keys() and "y" in data.keys():
        X_in = data["x"]
        Y_in = data["y"]
    elif "x_in" in data.keys() and "y_in" in data.keys():
        X_in = data["x_in"]
        Y_in = data["y_in"]
    else:
        raise KeyError("not find input data")

    #Create feature list
    d = X_in.shape[0]
    featname = [('%d' % i) for i in range(1,d+1)]

    return X_in, Y_in, featname
