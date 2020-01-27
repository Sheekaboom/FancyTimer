# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:37:05 2020

@author: aweis
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options



extensions = [Extension('pycom.aoa.base.AoaAlgorithmCython',['./pycom/aoa/base/AoaAlgorithmCython.pyx']),
              Extension('pycom.aoa.cython.CBFCython', ['./pycom/aoa/cython/CBFCython.pyx'])]

Options.annotate = True

setup(
    ext_modules = cythonize(extensions)
)


#build with `python setup.py build_ext --inplace` in anaconda prompt.
#Befoer running this file needs to be moved one level outside of the package
