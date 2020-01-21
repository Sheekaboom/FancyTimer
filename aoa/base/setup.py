# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:37:05 2020

@author: aweis
"""

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.annotate = True

setup(
    ext_modules = cythonize("AoaAlgorithmCython.pyx")
)


#build with `python setup.py build_ext --inplace` in anaconda prompt
