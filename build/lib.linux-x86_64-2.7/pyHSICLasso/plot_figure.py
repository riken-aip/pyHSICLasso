#!usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import range

from future import standard_library
from matplotlib import pyplot as plt

standard_library.install_aliases()


def plot_figure(path, beta, A):
    t = path.sum(0)
    plt.title("HSICLasso Result")
    plt.xlabel("lambda")
    plt.ylabel("cofficients")
    for ind in range(len(A)):
        plt.plot(t, path[A[ind], :], label="{}".format(A[ind] + 1))
    plt.legend()
    plt.show()
