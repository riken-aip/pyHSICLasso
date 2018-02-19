#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from builtins import open

from future import standard_library

from setuptools import setup

standard_library.install_aliases()



here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="pyHSICLasso",
    version="1.0.0",
    author="Makoto Yamada",
    author_email="makoto.yamada@riken.jp",
    url="http://www.makotoyamada-ml.com/hsiclasso.html",
    description="supervised feature selection considering the dependency of\
        nonlinear input and output.",
    long_description=long_description,
    download_url="https://github.com/suecharo/pyHSICLasso",
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"
        "Topic :: Scientific/Engineering",
    ),
    platforms=["python2.7", "python3.4", "python3.5", "python3.6"],
    license="MIT",
    keywords="HSIC Lasso HSICLasso feature-selection data-science",
    # install_requires=["numpy", "scipy", "matplotlib", "pandas", "future"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    packages=["pyHSICLasso"],
)
