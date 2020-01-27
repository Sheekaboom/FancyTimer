# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np
import scipy.linalg
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


#%% Some initialization
if DEBUG:
    mat_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32);
    solve_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32)
    num_reps = 5
else:
    mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
    solve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)
    num_reps = 25

out_file = os.path.join(output_directory,'stats_py_np_matrix.mat')

#%% Test the fancy_timeit_matrix_sweep capability
#lu_factor is used as opposed to lu to match the output of MATLAB's lu() function
functs      = [scipy.linalg.lu_factor,np.matmul]
funct_names = ['lu'                  ,'matmul' ]
num_args    = [1                     ,2     ]

#% now run 
[stats_np_double,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
[stats_np_single,_] = fancy_timeit_matrix_sweep(
	functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);

#%% and now matrix solving 
import scipy.linalg
solve_reps = num_reps

#argument generation functions
def solve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    return [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.cdouble),
             (np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)]
def solve_arg_gen_funct_single(dim,num_args): #generate A and b for dense solving
    return [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.csingle),
             (np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)]
			 
[solve_stats_double,_] = fancy_timeit_matrix_sweep(
        [scipy.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
        ,arg_gen_funct=solve_arg_gen_funct);
[solve_stats_single,_] = fancy_timeit_matrix_sweep(
        [scipy.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
        ,arg_gen_funct=solve_arg_gen_funct_single);
		
stats_np_double.update(solve_stats_double)
stats_np_single.update(solve_stats_single)

#%% now save out in case sparse does something stupid
import scipy.io as sio
sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})

    