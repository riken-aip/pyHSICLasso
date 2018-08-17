#!usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
from future import standard_library
from scipy import io as spio

standard_library.install_aliases()

def input_csv_file(file_name,output_list=['class']):
    df = pd.read_csv(file_name, sep=",")

    featname  = df.columns.tolist()
    input_index = list(range(0,len(featname)))
    output_index = []

    for output_name in output_list:
        if output_name in featname:
            tmp = featname.index(output_name)
            featname.remove(output_name)
            output_index.append(tmp)
            input_index.remove(tmp)
        else:
            raise ValueError("Output variable, %s, not found" % (output_name))


    X_in = df.ix[:, input_index].values.T

    if len(output_index) == 1:
        Y_in = df.ix[:, output_index].values.reshape(1, len(df.index))
    else:
        Y_in = df.ix[:, output_index].values.T

    return X_in, Y_in, featname


def input_tsv_file(file_name,output_list=['class']):
    df = pd.read_csv(file_name, sep="\t")

    # Store the column name (Feature name)
    featname = df.columns.tolist()
    input_index = list(range(0, len(featname)))
    output_index = []

    for output_name in output_list:
        if output_name in featname:
            tmp = featname.index(output_name)
            featname.remove(output_name)
            output_index.append(tmp)
            input_index.remove(tmp)
        else:
            raise ValueError("Output variable, %s, not found" % (output_name))

    X_in = df.ix[:, input_index].values.T

    if len(output_index) == 1:
        Y_in = df.ix[:, output_index].values.reshape(1, len(df.index))
    else:
        Y_in = df.ix[:, output_index].values.T

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
