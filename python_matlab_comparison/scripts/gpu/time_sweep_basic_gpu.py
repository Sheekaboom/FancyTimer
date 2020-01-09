# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np
import cupy as cp

#mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
#mat_dim_list = np.floor(np.linspace(1,5000,6)).astype(np.uint32);
mat_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32);
out_file_np = 'stats_py_cp_basic_gpu.mat'
#%%
'''
CUPY TESTS
'''
#%% argument generation
def cupy_arg_gen_funct(dim,num_args):
    cpu_data = [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.cdouble) for a in range(num_args)]
    gpu_data = [cp.array(cd) for cd in cpu_data]
    return gpu_data

def cupy_arg_gen_funct_single(dim,num_args):
    cpu_data = [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.csingle) for a in range(num_args)]
    gpu_data = [cp.array(cd) for cd in cpu_data]
    return gpu_data


#%% Test the fancy_timeit_matrix_sweep capability
combo_fun   = lambda x,y: x+y*np.exp(x)
functs      = [np.add,np.subtract,np.multiply,np.divide,np.exp,combo_fun]
funct_names = ['add' ,'sub'      ,'mult'     ,'div'    ,'exp' ,'combo'  ]
num_args    = [2     ,2          ,2          ,2        ,1     ,2        ]
num_reps    = 100

#% now run 
[stats_cp_double,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,
    arg_gen_funct=cupy_arg_gen_funct);
[stats_cp_single,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,
    arg_gen_funct=cupy_arg_gen_funct_single);

#%% now save out
import scipy.io as sio
sio.savemat(out_file_np,{'double':stats_cp_double,'single':stats_cp_single})


    