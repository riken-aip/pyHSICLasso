# pyHSICLasso
[![Build Status](https://travis-ci.org/riken-aip/pyHSICLasso.svg?branch=master)](https://travis-ci.org/riken-aip/pyHSICLasso)

pyHSICLasso is a supervised feature selection considering the dependency of nonlinear input and output.

## What can you do with this?
The goal of supervised feature selection is to find a subset of input features that are responsible for predicting output values. By using this, you can supplement the dependence of nonlinear input and output and you can calculate the optimal solution efficiently for high dimensional problem. The effectiveness are demonstrated through feature selection experiments for classification and regression with thousands of features. Finding a subset of features in high-dimensional supervised learning is an important problem with many real- world applications such as gene selection from microarray data, document categorization, and prosthesis control.

## Install
```sh
$ pip install -r requirements.txt
$ python setup.py install
```

or  

```sh
$ pip install pyHSICLasso
```

## Usage
First, pyHSICLasso provides the single entry point as class `HSICLasso()`

This class has the following methods.

- input
- regression
- classification
- dump
- plot_path
- plot_dendrogram
- get_features
- get_features_neighbors
- get_index
- get_index_neighbors
- get_index_neighbors_score

The input format corresponds to the following formats.

- MATLAB file (.mat)
- .csv
- .tsv
- python's list
- numpy's ndarray

When using .mat, .csv, .tsv, it is better to use pandas dataframe. 
The rows of the dataframe are  sample number. The first column is classification value. 
The remaining columns are values of each features. The following is a sample data (csv format). 

```
class,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10
-1,2,0,0,0,-2,0,-2,0,2,0
1,2,2,0,0,-2,0,0,0,2,0
...

```

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
============================================== HSICLasso : Result ==================================================
| Order | Feature     | Score | Top-5 Related Feature (Relatedness Score)                                          |
| 1     | v1423       | 1.000 | v493    (0.413), v1674   (0.384), v245    (0.384), v267    (0.384), v415    (0.346)|
| 2     | v513        | 0.765 | v365    (0.563), v1648   (0.487), v1139   (0.456), v1912   (0.450), v241    (0.446)|
| 3     | v249        | 0.679 | v267    (0.544), v245    (0.544), v822    (0.381), v824    (0.374), v1897   (0.343)|
| 4     | v1671       | 0.639 | v513    (0.231), v1263   (0.217), v1771   (0.202), v1912   (0.197), v187    (0.179)|
| 5     | v780        | 0.116 | v513    (0.439), v26     (0.439), v571    (0.410), v127    (0.369), v91     (0.361)|

>>> hsic_lasso.get_index()
[1422, 512, 248, 1670, 779]

>>> hsic_lasso.get_features()
['v1423', 'v513', 'v249', 'v1671', 'v780']

>>> hsic_lasso.get_index_neighbors(feat_index=0,num_neighbors=5)
[492, 1673, 244, 266, 414]

>>> hsic_lasso.get_features_neighbors(feat_index=0,num_neighbors=5)
['v493', 'v1674', 'v245', 'v267', 'v415']

>>> hsic_lasso.get_index_neighbors_score(feat_index=0,num_neighbors=5)
[ 0.412915 ,  0.38446  ,  0.38446  ,  0.38446  ,  0.3462652]



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
