#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst")) as f:
    long_description = f.read()


def _load_requires_from_file(filepath):
    return [pkg_name.rstrip('\r\n') for pkg_name in open(filepath).readlines()]


def _install_requires():
    return _load_requires_from_file('requirements.txt')


def _packages():
    return find_packages(
        exclude=[
            '*.tests',
            '*.tests.*',
            'tests.*',
            'tests'
        ]
    )


setup(
    name="pyHSICLasso",
    version="1.4.2",
    author="Makoto Yamada",
    author_email="makoto.yamada@riken.jp",
    url="http://www.makotoyamada-ml.com/hsiclasso.html",
    description="Supervised, nonlinear feature selection method for high-dimensional datasets.",
    long_description=long_description,
    download_url="https://github.com/riken-aip/pyHSICLasso",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    platforms=["python2.7", "python3.4", "python3.5", "python3.6"],
    license="MIT",
    keywords="HSIC Lasso HSICLasso feature-selection feature-extraction machine-learning nonlinear data-science",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=_install_requires(),
    packages=_packages()
)
