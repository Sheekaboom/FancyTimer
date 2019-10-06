# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:43:32 2019

@author: aweiss
"""

from pycom.beamforming.python_interface.BeamformTest import fancy_timeit

import numpy as np
import scipy.linalg as linalg
from scipy.linalg import lu as lu
import numba
import pycuda

m = 5000; n = 5000
num_reps = 1
dtype = np.double
a = np.random.rand(m,n)+1j*np.random.rand(m,n)
b = np.random.rand(m,n)+1j*np.random.rand(m,n)
a = a.astype(dtype)
b = b.astype(dtype)

def display_time_stats(time_data,name):
    '''
    @brief print our time data from our Function
    '''
    print('{} :'.format(name))
    for k,v in time_data.items():
        print('    {} : {}'.format(k,v))
        
        
'''
THios code is copied from https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en
'''


'''
end copied code
'''


'''
Add/Sub/Mult/Div
'''
add_fun = lambda : np.add(a,b)
[add_double_complex_time,add_double_complex_rv] = fancy_timeit(add_fun,num_reps);
display_time_stats(add_double_complex_time,'Add Complex Double')
'''
sub_fun = lambda : np.subtract(a,b);
[sub_double_complex_time,sub_double_complex_rv] = fancy_timer(sub_fun,num_reps);
mul_fun = lambda : np.multiply(a,b);
[mul_double_complex_time,mul_double_complex_rv] = fancy_timer(mul_fun,num_reps);
div_fun = lambda : np.divide(a,b);
[div_double_complex_time,div_double_complex_rv] = fancy_timer(div_fun,num_reps);
'''




'''
exp/matmul/sum
'''
matmul_fun = lambda : np.matmul(a,b)
[matmul_double_complex_time,matmul_double_complex_rv] = fancy_timeit(matmul_fun,num_reps);
display_time_stats(matmul_double_complex_time,'Matrix Multiply Complex Double')


'''
LU Decomposition
'''
lu_fun = lambda : lu(a)
[lu_double_complex_time,lu_double_complex_rv] = fancy_timeit(lu_fun,num_reps);
display_time_stats(lu_double_complex_time,'LU Complex Double')







