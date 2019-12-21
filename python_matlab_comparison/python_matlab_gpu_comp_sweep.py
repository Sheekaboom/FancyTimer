# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:03:48 2019

@author: aweiss
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""

#import skcuda.cublas

from pycom.beamforming.python_interface.OperationTimer import fancy_timeit_matrix_sweep
import numpy as np

#%% Some initialization
#%mat_dim_list = 2.^(4:14);
mydtype = np.cdouble
mat_dim_list = (np.floor(np.linspace(0,10000,51)).astype(np.uint32))[1:];
#mat_dim_list = (np.floor(np.linspace(0,1000,5)).astype(np.uint32))[1:];
#mat_dim_list = np.floor(np.logspace(1,3,100)).astype(np.uint32);

#%%
'''
PYCUDA/SKCUDA TESTS
'''
'''
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import skcuda.linalg
import skcuda.misc
#import skcuda.cula
#%% init our lists for running
funct_lists_pc   = []; 
funct_namess_pc  = []; 
num_arg_lists_pc = []; 
rv_pc            = []; 
stats_pc         = []; 
num_reps_pc      = []; 

def pycuda_arg_gen_funct(dim,num_args):
    return [pycuda.gpuarray.to_gpu((np.random.rand(dim,dim).astype(np.cdouble)+1j*np.random.rand(dim,dim).astype(np.cdouble)))
                                                            for a in range(num_args)]

#%% Test the fancy_timeit_matrix_sweep capability
#%add/sub/mult/div
funct_lists_pc  .append([skcuda.misc.add,skcuda.misc.subtract,skcuda.misc.multiply,skcuda.misc.divide]);
funct_namess_pc .append(['add'          ,'sub'               ,'mult'              ,'div'             ]);
num_arg_lists_pc.append([2              ,2                   ,2                   ,2                 ]);
num_reps_pc     .append(100)

#%exp/sum/combo
comfun_pc = lambda a,b: a*b+pycuda.cumath.exp(a); #%combined operations
#comfun_pc = pycuda.elementwise.ElementwiseKernel('pycuda::complex<double> *a,pycuda::complex<double> *b',
#                                                 'a*b+exp(a)','combined_funct')
funct_lists_pc  .append([pycuda.cumath.exp,pycuda.gpuarray.sum,comfun_pc]);
funct_namess_pc .append(['exp'            ,'sum'              ,'combo'  ]);
num_arg_lists_pc.append([1     ,1     ,2        ]);
num_reps_pc     .append(100);

#%LU/matmul
from skcuda.cublas import cublasCreate
skcuda_cublas_handle = cublasCreate()
def skcuda_dot(*args,**kwargs):
    #quick wrapper for skucda dot product
    out_arr = pycuda.gpuarray.GPUArray(args[0].shape,args[0].dtype)
    skcuda.linalg.dot(*args,out=out_arr,handle=skcuda_cublas_handle)
    return out_arr
    
#from scipy.linalg import lu
    #LU was removed here. Should use cula Dense toolkit but their web download page seems to be broken. The project may be abandoned
#funct_lists_pc  .append([skcuda.cula.culaDeviceZgetrf,skcuda_dot]);
#funct_namess_pc .append(['lu'                        ,'matmul'  ]);
#num_arg_lists_pc.append([1                           ,2         ]);
funct_lists_pc  .append([skcuda_dot]);
funct_namess_pc .append(['matmul'  ]);
num_arg_lists_pc.append([2         ]);
num_reps_pc     .append(25);

#num_reps_pc = [1,1,1]

#% now run everything
for i in range(len(funct_lists_pc)):
    funct_list   = funct_lists_pc[i];
    funct_names  = funct_namess_pc[i];
    num_arg_list = num_arg_lists_pc[i];
    reps         = num_reps_pc[i];
    [cur_st,cur_rv] = fancy_timeit_matrix_sweep(
        funct_list,funct_names,num_arg_list,mat_dim_list,reps
        ,arg_gen_funct=pycuda_arg_gen_funct);
    rv_pc = cur_rv
    stats_pc.append(cur_st)
'''
#%% now measure our fft
'''
# CURENTLY NOT WORKING. cudaErrorInsufficientDriver Error raised
import skcuda.fft
fft_dim_list = np.floor(np.linspace(1,5000000,51)).astype(np.uint32)
fft_reps = 100

def fft_funct_pc(a_gpu):
    #wrap skcuda.fft.fft function
    plan = skcuda.fft.Plan((a_gpu.shape),mydtype,mydtype)
    out = pycuda.gpuarray.empty_like(a_gpu)
    skcuda.fft.fft(a_gpu,out,plan)
    return out

def fft_arg_gen_funct(dim,num_args): #vector generation function 
        return [pycuda.gpuarray.to_gpu((np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble))]
    
[fft_st,fft_rv] = fancy_timeit_matrix_sweep(
        [fft_funct_pc],['fft'],[1],fft_dim_list,fft_reps
        ,arg_gen_funct=fft_arg_gen_funct);
rv_pc.append(fft_rv)
stats_pc.append(fft_st)
'''

#%% and now matrix solving 
#TODO update this to pycuda/skcuda Only has cho_solve
#this gives CUSOLVER_STATUS_NOT_INITIALIZED error and is poorly documented
'''
import skcuda.linalg
solve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)
solve_reps = 25
solve_reps = 1

def pc_cho_solve(A,b):
    cA = skcuda.linalg.cho_factor(A)
    return skcuda.linalg.cho_solve(cA,b)
    
def pc_solve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    return [pycuda.gpuarray.to_gpu((np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(mydtype)),
             pycuda.gpuarray.to_gpu((np.random.rand(dim)+1j*np.random.rand(dim)).astype(mydtype))]
[solve_st,solve_rv] = fancy_timeit_matrix_sweep(
        [pc_cho_solve],['solve'],[2],solve_dim_list,solve_reps
        ,arg_gen_funct=pc_solve_arg_gen_funct);
rv_pc.append(solve_rv)
stats_pc.append(solve_st)
'''

#%%
'''
CUPY TESTS
'''
import cupy as cp
import cupyx.scipy.linalg
#%% init lists
funct_lists_cp   = []; 
funct_namess_cp  = []; 
num_arg_lists_cp = []; 
rv_cp            = []; 
stats_cp         = []; 
num_reps_cp      = []; 

def cupy_cleanup_funct(arg_list):
    '''@brief clear all cupy memory on the gpu'''
    #pinned_mempool = cp.get_default_pinned_memory_pool()
    #pinned_mempool.free_all_blocks()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    #print("    USED MEM : {}   ".format(mempool.used_bytes()))
    #print("    FREE MEM : {}   ".format(mempool.total_bytes()))
    #print("FREE PMEM : {}   ".format(pinned_mempool.n_free_blocks()),end='')
    #print("Freed Memory (supposedly)")

def cupy_arg_gen_funct(dim,num_args):
    cpu_data = [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)) for a in range(num_args)]
    gpu_data = [cp.array(cd) for cd in cpu_data]
    #data = [(cp.random.rand(dim,dim)+1j*cp.random.rand(dim,dim)) for a in range(num_args)]
    mempool = cp.get_default_memory_pool()
    print('   FREE: {}'.format(mempool.free_bytes()))
    print('   USED: {}'.format(mempool.used_bytes()))
    print('  TOTAL: {}'.format(mempool.total_bytes()))
    return gpu_data

#%% GPU Data move to GPU
funct_lists_cp  .append([cp.add,cp.subtract,cp.multiply,cp.divide]);
funct_namess_cp .append(['add' ,'sub'      ,'mult'     ,'div'    ]);
num_arg_lists_cp.append([2     ,2          ,2          ,2        ]);
num_reps_cp     .append(100)

#%exp/sum/combo
#comfun_cp_elemwise = cp.ElementwiseKernel('complex128 a,complex128 b',
#                                          'complex128 c',
#                                          'c=a*b+exp(a)','combined_funct')
@cp.fuse()
def comfun_cp(a,b):
    a*b+cp.exp(a); #%combined operations cp.ElementwiseKernel(???)
#funct_lists_cp  .append([cp.exp,cp.sum,comfun_cp,comfun_cp_elemwise]);
#funct_namess_cp .append(['exp' ,'sum' ,'combo'  ,'combo_elemwise'  ]);
#num_arg_lists_cp.append([1     ,1     ,2        ,2                 ]);
#num_reps_cp     .append(100);
funct_lists_cp  .append([cp.exp,cp.sum,comfun_cp]);
funct_namess_cp .append(['exp' ,'sum' ,'combo'  ]);
num_arg_lists_cp.append([1     ,1     ,2        ]);
num_reps_cp     .append(100);

#%LU/matmul
#from scipy.linalg import lu
#funct_lists_cp  .append([cupyx.scipy.linalg.lu_factor,cp.matmul]); #LU complex not supported
#funct_namess_cp .append(['lu'                        ,'matmul' ]);
funct_lists_cp  .append([cp.matmul]);
funct_namess_cp .append(['matmul'  ]);
num_arg_lists_cp.append([2        ]);
num_reps_cp     .append(25);

#num_reps_cp = [1,1,1]

#% now run everything
for i in range(len(funct_lists_cp)):
    funct_list   = funct_lists_cp[i];
    funct_names  = funct_namess_cp[i];
    num_arg_list = num_arg_lists_cp[i];
    reps         = num_reps_cp[i];
    [cur_st,cur_rv] = fancy_timeit_matrix_sweep(
        funct_list,funct_names,num_arg_list,mat_dim_list,reps
        ,arg_gen_funct=cupy_arg_gen_funct,cleanup_funct= cupy_cleanup_funct);
    rv_cp = cur_rv
    stats_cp.append(cur_st)
    
#%% now measure our fft
fft_dim_list = np.floor(np.linspace(1,5000000,51)).astype(np.uint32)
fft_dim_list = np.floor(np.linspace(1,5000,5)).astype(np.uint32)
fft_reps = 100
#fft_reps = 1

def fft_arg_gen_funct(dim,num_args): #vector generation function 
        cpu_data = [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble) for a in range(num_args)]
        gpu_data = [cp.array(cd) for cd in cpu_data]
        mempool = cp.get_default_memory_pool()
        print('   FREE: {}'.format(mempool.free_bytes()))
        print('   USED: {}'.format(mempool.used_bytes()))
        print('  TOTAL: {}'.format(mempool.total_bytes()))
        return gpu_data
    
[fft_st,fft_rv] = fancy_timeit_matrix_sweep(
        [cp.fft.fft],['fft'],[1],fft_dim_list,fft_reps
        ,arg_gen_funct=fft_arg_gen_funct,cleanup_funct= cupy_cleanup_funct);
rv_cp = fft_rv
stats_cp.append(fft_st)

#%% and now matrix solving 
#TODO update this to cupy
import cupy.linalg
solve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)
solve_reps = 25
#solve_reps = 1
def cp_solve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
    cpu_data = [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(mydtype),
             (np.random.rand(dim)+1j*np.random.rand(dim)).astype(mydtype)]
    gpu_data = [cp.array(cd) for cd in cpu_data]
    mempool = cp.get_default_memory_pool()
    print('   FREE: {}'.format(mempool.free_bytes()))
    print('   USED: {}'.format(mempool.used_bytes()))
    print('  TOTAL: {}'.format(mempool.total_bytes()))
    return gpu_data
[solve_st,solve_rv] = fancy_timeit_matrix_sweep(
        [cp.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
        ,arg_gen_funct=cp_solve_arg_gen_funct,cleanup_funct= cupy_cleanup_funct);
        
stats_cp.append(solve_st)


#%% now save out
import scipy.io as sio
#sio.savemat('stats_py_gpu.mat',{'stats_py_np':stats_pc,'stats_py_nb':stats_cp})
sio.savemat('stats_py_gpu.mat',{'stats_py_nb':stats_cp})
    