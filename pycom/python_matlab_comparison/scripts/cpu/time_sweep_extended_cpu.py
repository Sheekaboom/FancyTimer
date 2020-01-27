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
    
# what directory to store the data in
try:
    from __main__ import output_directory
except:
    output_directory = './data'


#%% Some initialization
if DEBUG:
    mat_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32);
    fft_dim_list = np.floor(np.linspace(1,50000,6)).astype(np.uint32)
    num_reps = 10
else:
    mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
    fft_dim_list = np.floor(np.linspace(1,5000000,51)).astype(np.uint32)
    num_reps = 100
    
out_file = os.path.join(output_directory,'stats_py_np_extended.mat')

#%%
'''
NUMPY TESTS
'''
#%% Test the fancy_timeit_matrix_sweep capability
functs      = [np.sum]
funct_names = ['sum' ]
num_args    = [1     ]

#% now run 
[stats_np_double,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
[stats_np_single,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);
	
	
#%% now measure our fft
fft_reps = num_reps

#%% generate our arguments
def fft_arg_gen_funct(dim,num_args): #vector generation function 
        return [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)
                                    for a in range(num_args)]
# generate our arguments
def fft_arg_gen_funct_single(dim,num_args): #vector generation function 
        return [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)
                                    for a in range(num_args)]

#%% and run 
[stats_fft_double,_] = fancy_timeit_matrix_sweep(
        [np.fft.fft],['fft'],[1],fft_dim_list,fft_reps
        ,arg_gen_funct=fft_arg_gen_funct);
stats_np_double.update(stats_fft_double)

[stats_fft_single,_] = fancy_timeit_matrix_sweep(
        [np.fft.fft],['fft'],[1],fft_dim_list,fft_reps
        ,arg_gen_funct=fft_arg_gen_funct_single);
stats_np_single.update(stats_fft_single)

    
#%% now save out
import scipy.io as sio
sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})
    
	
	
	
	