# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:58:19 2019

@author: aweis
"""
from pycom.beamforming.python_interface.BeamformTest import fancy_timeit
from numba import vectorize
import numpy as np
import cmath
#implement basic math operations in numpy, numba, fortran?
num_reps = 5
m = 10000; n=10000
mydtype = np.double
#numba_setup = '''
from numba import vectorize
import numpy as np
#define multiplicaltion
@vectorize(['complex128(complex128,complex128)'],target='parallel')
def vector_mult_complex(a,b):
     return a+a*b*cmath.exp(a)
#create our values
vals_a = np.random.rand(m,n).astype(mydtype)+1j*np.random.rand(m,n).astype(mydtype)
vals_b = np.random.rand(m,n).astype(mydtype)+1j*np.random.rand(m,n).astype(mydtype)
compile_run = vector_mult_complex(vals_a[0,0],vals_b[0,0])
print("Setup Complete")
#'''.format(m=m,n=n,vtype="complex128",dtype='np.cdouble')
# numba_loop = '''
#rv = vector_mult_complex(vals_a,vals_b)
#'''
#mult_time_numba = timeit.timeit(stmt=numba_loop,setup=numba_setup,number=10)
mult_time_numba,rv = fancy_timeit(lambda:vector_mult_complex(vals_a,vals_b),num_reps=1)
print(mult_time_numba.tostring())






