# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np
import os

#this allows setting of a global debug
try:
    from __main__ import DEBUG
    if DEBUG is None:
        raise Exception("DEBUG is None")
except:
    DEBUG = False #debug mode for testing
    print("DEBUG Not defined. Setting to False")
if DEBUG:
    print("Debugging Mode Enabled")
    
# what directory to store the data in
try:
    from __main__ import output_directory
except:
    output_directory = './data'


#%% intialize sweep size and repeats
if DEBUG:
    mat_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32);
    num_reps = 10
else:
    mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
    num_reps = 100

out_file_np = os.path.join(output_directory,'stats_py_np_basic.mat')
out_file_nb = os.path.join(output_directory,'stats_py_nb_basic.mat')

#%%
'''
NUMPY TESTS
'''
#%% Test the fancy_timeit_matrix_sweep capability
combo_fun   = lambda x,y: x+y*np.exp(x)
functs      = [np.add,np.subtract,np.multiply,np.divide,np.exp,combo_fun]
funct_names = ['add' ,'sub'      ,'mult'     ,'div'    ,'exp' ,'combo'  ]
num_args    = [2     ,2          ,2          ,2        ,1     ,2        ]

#% now run 
[stats_np_double,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
[stats_np_single,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);

#%%
'''
NUMBA TESTS
'''
#%% Function declarations
import cmath
from numba import vectorize,complex128,complex64

@vectorize([complex64(complex64,complex64),complex128(complex128,complex128)],target='parallel')
def nb_add(x,y):
    return x+y

@vectorize([complex64(complex64,complex64),complex128(complex128,complex128)],target='parallel')
def nb_subtract(x,y):
    return x+y

@vectorize([complex64(complex64,complex64),complex128(complex128,complex128)],target='parallel')
def nb_multiply(x,y):
    return x+y

@vectorize([complex64(complex64,complex64),complex128(complex128,complex128)],target='parallel')
def nb_divide(x,y):
    return x+y

@vectorize([complex64(complex64),complex128(complex128)],target='parallel')
def nb_exp(x):
    return cmath.exp(x)

@vectorize([complex64(complex64,complex64),complex128(complex128,complex128)],target='parallel')
def nb_comb(x,y):
    return x+y*cmath.exp(x)

#%% Test the fancy_timeit_matrix_sweep capability
#%add/sub/mult/div
functs       = [nb_add,nb_subtract,nb_multiply,nb_divide,nb_exp,nb_comb];
funct_names  = ['add' ,'sub'      ,'mult'     ,'div'    ,'exp' ,'combo'];
num_args     = [2     ,2          ,2          ,2        ,1     ,2      ];

#% now run
[stats_nb_double,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble); 
[stats_nb_single,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle); 
    
#%% now save out
import scipy.io as sio
sio.savemat(out_file_np,{'double':stats_np_double,'single':stats_np_single})
sio.savemat(out_file_nb,{'double':stats_nb_double,'single':stats_nb_single})

    