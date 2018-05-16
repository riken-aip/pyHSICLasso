# pyHSICLasso
[![Build Status](https://travis-ci.org/suecharo/pyHSICLasso.svg?branch=master)](https://travis-ci.org/suecharo/pyHSICLasso)

pyHSICLasso is a supervised feature selection considering the dependency of nonlinear input and output.

## What can you do with this?
The goal of supervised feature selection is to find a subset of input features that are responsible for predicting output values. By using this, you can supplement the dependence of nonlinear input and output and you can calculate the optimal solution efficiently for high dimensional problem. The effectiveness are demonstrated through feature selection experiments for classification and regression with thousands of features. Finding a subset of features in high-dimensional supervised learning is an important problem with many real- world applications such as gene selection from microarray data, document categorization, and prosthesis control.

## Install
```sh
$ pip install numpy scipy pandas matplotlib future
$ pip install pyHSICLasso
```

## Usage
First, pyHSICLasso provides the single entry point as class `HSICLasso()`

This class has the following methods.

- input
- regression
- classification
- dump
- plot
- get_index

The input format corresponds to the following formats.

- MATLAB file (.mat)
- .csv
- .tsv
- python's list
- numpy's ndarray

When using .mat, .csv, .tsv, it is better to use pandas dataframe. The rows of the dataframe are  sample number. The first column is classification value. The remaining columns are values of each features.

When using python's list or numpy's ndarray, Let each index be sample number, let values of each features for X[ind] and classification value for Y[ind].

```py
>>> from pyHSICLasso import HSICLasso
>>> hsic_lasso = HSICLasso()

>>> hsic_lasso.input("data.mat")

>>> hsic_lasso.input("data.csv")

>>> hsic_lasso.input("data.tsv")

>>> hsic_lasso.input([[1, 1, 1], [2, 2, 2]], [0, 1])

>>> hsic_lasso.input(np.array([[1, 1, 1], [2, 2, 2]]), np.array([0, 1]))
```

You can specify the number of subset of feature selections with arguments `regression` and` classification`.

```py
>>> hsic_lasso.regression(5)

>>> hsic_lasso.classification(10)
```

About output method, it is possible to select plots on the graph, details of the analysis result, output of the feature index.

```py
>>> hsic_lasso.plot()
# plot the graph

>>> hsic_lasso.dump()
===== HSICLasso : Result ======
| Order | Feature | Score |
| 1     | 1099    | 0.159 |
| 2     | 99      | 0.034 |
| 3     | 1574    | 0.024 |
| 4     | 1645    | 0.006 |
| 5     | 299     | 0.001 |
===== HSICLasso : Path ======
[ 0.02513268  0.02980639  0.03351551  0.03495434  0.03539473]
[ 0.          0.00470815  0.00851197  0.01000344  0.01047563]
[ 0.          0.          0.00627558  0.00866519  0.00941868]
[ 0.          0.          0.          0.00244106  0.00319874]
[ 0.          0.          0.          0.          0.00085114]

>>> hsic_lasso.get_index()
[1099, 99, 1574, 1645, 299]
```
![graph](https://www.fastpic.jp/images.php?file=6530104232.png)


## Contributors
### Auther
Name : Makoto Yamada

E-mail : makoto.yamada@riken.jp

- [HSICLasso Page](http://www.makotoyamada-ml.com/hsiclasso.html)
- [HSICLasso Paper](https://arxiv.org/pdf/1202.0515.pdf)

### Distributor
Name : Hirotaka Suetake

E-mail : hirotaka.suetake@riken.jp
