#!usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt

def plot_figure(path, beta, A):
    t = path.sum(0)
    plt.title("HSICLasso Result")
    plt.xlabel("lambda")
    plt.ylabel("cofficients")
    for ind in range(len(A)):
        plt.plot(t, path[A[ind], :], label="{}".format(A[ind] + 1))
    plt.legend()
    plt.show()
