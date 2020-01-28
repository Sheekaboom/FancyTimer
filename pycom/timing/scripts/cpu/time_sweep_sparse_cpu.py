# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

from pycom.timing.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np
import scipy.linalg
import os

#this allows setting of a global debug
try:
    from __main__ import DEBUG
    if DEBUG is None:
        raise Exception("DEBUG is None")
except Exception as e:
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
    ssolve_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32)[1:]
    num_reps = 5
else:
    ssolve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)[1:]
    num_reps = 25

out_file = os.path.join(output_directory,'stats_py_np_sparse.mat')

#%% and sparse solving
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

ssolve_reps = num_reps

def generate_random_sparse(shape,num_el,dtype):
    '''@brief fill a sparse array with numel_random elements'''
    data = (np.random.rand(num_el)+1j*np.random.rand(num_el)).astype(dtype)
    ri = np.random.randint(0,high=shape[0]-1,size=num_el)
    ci = np.random.randint(0,high=shape[1]-1,size=num_el)
    rv = scipy.sparse.csr_matrix((data,(ri,ci)),shape=shape,dtype=dtype)
    return rv  

#argument generation
def ssolve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    numel = np.floor(dim*5).astype(np.uint32)
    return [generate_random_sparse((dim,dim),numel,np.cdouble),(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)]  
def ssolve_arg_gen_funct_single(dim,num_args): #generate A and b for dense solving
    numel = np.floor(dim*5).astype(np.uint32)
    return [generate_random_sparse((dim,dim),numel,np.csingle),(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)]  

#now get the results
[ssolve_stats_double,_] = fancy_timeit_matrix_sweep(
        [scipy.sparse.linalg.spsolve],['ssolve'],[2],ssolve_dim_list,ssolve_reps
        ,arg_gen_funct=ssolve_arg_gen_funct,no_fail=True);
#now get the results
[ssolve_stats_single,_] = fancy_timeit_matrix_sweep(
        [scipy.sparse.linalg.spsolve],['ssolve'],[2],ssolve_dim_list,ssolve_reps
        ,arg_gen_funct=ssolve_arg_gen_funct_single,no_fail=True);
		
stats_np_double = ssolve_stats_double
stats_np_single = ssolve_stats_single
    
#%% now save out
import scipy.io as sio
sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})
    