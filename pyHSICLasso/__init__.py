#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from .api import HSICLasso
from .hsic_lasso import hsic_lasso
from .input_data import input_csv_file, input_tsv_file, input_matlab_file
standard_library.install_aliases()
