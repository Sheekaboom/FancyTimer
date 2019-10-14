# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:31:34 2019

@author: aweiss
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_test.pyx")
)