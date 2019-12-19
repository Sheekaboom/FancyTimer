# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

from pycom.beamforming.python_interface.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np

#%% Some initialization
#%mat_dim_list = 2.^(4:14);
#mat_dim_list = np.floor(np.logspace(1,4,100)).astype(np.uint32);
#mat_dim_list = np.floor(np.logspace(1,3,100)).astype(np.uint32);
mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
out_file = 'solve_stats_py.mat'
mydtype = np.cdouble
#%%
'''
NUMPY TESTS
'''
#%% init our lists for running
funct_lists_np   = []; 
funct_namess_np  = []; 
num_arg_lists_np = []; 
rv_np            = []; 
stats_np         = []; 
num_reps_np      = []; 

'''
#%% Test the fancy_timeit_matrix_sweep capability
#%add/sub/mult/div
funct_lists_np  .append([np.add,np.subtract,np.multiply,np.divide]);
funct_namess_np .append(['add' ,'sub'      ,'mult'     ,'div'   ]);
num_arg_lists_np.append([2     ,2          ,2          ,2       ]);
num_reps_np     .append(100)

#%exp/sum/combo
comfun_np = lambda a,b: a*b+np.exp(a); #%combined operations
funct_lists_np  .append([np.exp,np.sum,comfun_np]);
funct_namess_np .append(['exp' ,'sum' ,'combo'  ]);
num_arg_lists_np.append([1     ,1     ,2        ]);
num_reps_np     .append(100);

#%LU/matmul
from scipy.linalg import lu
funct_lists_np  .append([lu  ,np.matmul]);
funct_namess_np .append(['lu','matmul' ]);
num_arg_lists_np.append([1   ,2        ]);
num_reps_np     .append(25);

#num_reps_np = [1,1,1]

#% now run everything
for i in range(len(funct_lists_np)):
    funct_list   = funct_lists_np[i];
    funct_names  = funct_namess_np[i];
    num_arg_list = num_arg_lists_np[i];
    reps         = num_reps_np[i];
    [cur_st,cur_rv] = fancy_timeit_matrix_sweep(
        funct_list,funct_names,num_arg_list,mat_dim_list,reps);
    rv_np.append(cur_rv)
    stats_np.append(cur_st)

#%% now measure our fft
fft_dim_list = np.floor(np.linspace(1,5000000,51)).astype(np.uint32)
fft_reps = 100

def fft_arg_gen_funct(dim,num_args): #vector generation function 
        return [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(mydtype)
                                    for a in range(num_args)]
    
[fft_st,fft_rv] = fancy_timeit_matrix_sweep(
        [np.fft.fft],['fft'],[1],fft_dim_list,fft_reps
        ,arg_gen_funct=fft_arg_gen_funct);
rv_np.append(fft_rv)
stats_np.append(fft_st)
'''

#%% and now matrix solving 

import scipy.linalg
solve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)
solve_reps = 25
def solve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    return [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(mydtype),
             (np.random.rand(dim)+1j*np.random.rand(dim)).astype(mydtype)]
[solve_st,solve_rv] = fancy_timeit_matrix_sweep(
        [scipy.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
        ,arg_gen_funct=solve_arg_gen_funct);
rv_np.append(solve_rv)
stats_np.append(solve_st)

#%% and sparse solve SKIP THIS we can test this in FDFD
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

ssolve_dim_list = np.concatenate(([2],np.linspace(0,5000,51)[1::2])).astype(np.uint32)
ssolve_reps = 25

def generate_random_sparse(shape,num_el,dtype):
    '''
    @brief fill a sparse array with numel_random elements
    '''
    data = (np.random.rand(num_el)+1j*np.random.rand(num_el)).astype(dtype)
    ri = np.random.randint(0,high=shape[0]-1,size=num_el)
    ci = np.random.randint(0,high=shape[1]-1,size=num_el)
    rv = scipy.sparse.csr_matrix((data,(ri,ci)),shape=shape,dtype=dtype)
    return rv  

def ssolve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    numel = np.floor(dim*5).astype(np.uint32)
    return [generate_random_sparse((dim,dim),numel,mydtype),(np.random.rand(dim)+1j*np.random.rand(dim)).astype(mydtype)]  
[ssolve_st,ssolve_rv] = fancy_timeit_matrix_sweep(
        [scipy.sparse.linalg.spsolve],['ssolve'],[2],ssolve_dim_list,ssolve_reps
        ,arg_gen_funct=ssolve_arg_gen_funct);
rv_np.append(ssolve_rv)
stats_np.append(ssolve_st)
    
#%%
'''
NUMBA TESTS
'''
'''
#%% init our lists for running
funct_lists_nb   = []; 
funct_namess_nb  = []; 
num_arg_lists_nb = []; 
rv_nb            = []; 
stats_nb         = []; 
num_reps_nb      = []; 

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
funct_lists_nb  .append([nb_add,nb_subtract,nb_multiply,nb_divide]);
funct_namess_nb .append(['add' ,'sub'      ,'mult'     ,'div'   ]);
num_arg_lists_nb.append([2     ,2          ,2          ,2       ]);
num_reps_nb     .append(100)

#%exp/sum/combo
comfun_nb = lambda a,b: a*b+np.exp(a); #%combined operations
funct_lists_nb  .append([nb_exp,nb_comb  ]);
funct_namess_nb .append(['exp' ,'combo'  ]);
num_arg_lists_nb.append([1     ,2        ]);
num_reps_nb     .append(100);

#num_reps_nb = [1,1]

#% now run everything
for i in range(len(funct_lists_nb)):
    funct_list   = funct_lists_nb[i];
    funct_names  = funct_namess_nb[i];
    num_arg_list = num_arg_lists_nb[i];
    reps         = num_reps_nb[i];
    [cur_st,cur_rv] = fancy_timeit_matrix_sweep(
        funct_list,funct_names,num_arg_list,mat_dim_list,reps);
    rv_nb.append(cur_rv)
    stats_nb.append(cur_st)  
'''
stats_nb = [0]
    
#%% now save out
import scipy.io as sio
sio.savemat(out_file,{'stats_py_np':stats_np,'stats_py_nb':stats_nb})
    