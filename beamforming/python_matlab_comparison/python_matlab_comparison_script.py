# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:43:32 2019

@author: aweiss
"""

#%% Imports
from pycom.beamforming.python_interface.BeamformTest import fancy_timeit

import numpy as np
import scipy.linalg as linalg
from scipy.linalg import lu
from collections import OrderedDict
#import pycuda

ts_double = OrderedDict() #output structure for times

def display_time_stats(time_data,name):
    '''
    @brief print our time data from our Function
    '''
    print('{} :'.format(name))
    for k,v in time_data.items():
        print('    {} : {}'.format(k,v))
    return time_data    

#%% Some other things
out_file_name = 'python_complex_single_times.json'
m = 5000; n = 5000
num_reps = 100
dtype = np.csingle
np.random.seed(1234)
a = np.random.rand(m,n)+1j*np.random.rand(m,n)
b = np.random.rand(m,n)+1j*np.random.rand(m,n)
a = a.astype(dtype)
b = b.astype(dtype)    
        
'''
THios code is copied from https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization?lang=en
'''


'''
end copied code
'''

#%%
'''
Add/Sub/Mult/Div
'''
add_fun = lambda : np.add(a,b)
[add_double_complex_time,add_double_complex_rv] = fancy_timeit(add_fun,num_reps);
ts_double['add'] = display_time_stats(add_double_complex_time,'Add Complex Double')
sub_fun = lambda : np.subtract(a,b);
[sub_double_complex_time,sub_double_complex_rv] = fancy_timeit(sub_fun,num_reps);
ts_double['sub'] = display_time_stats(sub_double_complex_time,'Sub Complex Double')
mul_fun = lambda : np.multiply(a,b);
[mul_double_complex_time,mul_double_complex_rv] = fancy_timeit(mul_fun,num_reps);
ts_double['mul'] = display_time_stats(mul_double_complex_time,'Mul Complex Double')
div_fun = lambda : np.divide(a,b);
[div_double_complex_time,div_double_complex_rv] = fancy_timeit(div_fun,num_reps);
ts_double['div'] = display_time_stats(div_double_complex_time,'Div Complex Double')

#%%
'''
exp/matmul/sum
'''
matmul_fun = lambda : np.matmul(a,b)
[matmul_double_complex_time,matmul_double_complex_rv] = fancy_timeit(matmul_fun,num_reps);
ts_double['matmul'] = display_time_stats(matmul_double_complex_time,'Matrix Multiply Complex Double')

exp_fun = lambda : np.exp(a);
[exp_double_complex_time,exp_double_complex_rv] = fancy_timeit(exp_fun,num_reps);
ts_double['exp'] = display_time_stats(exp_double_complex_time,'Exponential Complex Double')

sum_fun = lambda : np.sum(a);
[sum_double_complex_time,sum_double_complex_rv] = fancy_timeit(sum_fun,num_reps);
ts_double['sum'] = display_time_stats(sum_double_complex_time,'Sum Complex Double')

#%%
'''
LU Decomposition
'''
lu_fun = lambda : lu(a)
[lu_double_complex_time,lu_double_complex_rv] = fancy_timeit(lu_fun,num_reps);
ts_double['lu'] = display_time_stats(lu_double_complex_time,'LU Complex Double')

#%%
'''
FFT
'''
np.random.seed(1234)
fft_n = 2**23; #number of points in our fft (same as matlab bench)
fft_data = np.random.rand(fft_n)+1j*np.random.rand(fft_n);
fft_data = fft_data.astype(dtype)
fft_fun = lambda : np.fft.fft(fft_data);
[fft_double_complex_time,fft_double_complex_rv] = fancy_timeit(fft_fun,num_reps);
ts_double['fft'] = display_time_stats(fft_double_complex_time,'FFT Complex Double')

#%%
'''
combined operation
'''
comb_fun = lambda : a*b+np.exp(a)
[comb_double_complex_time,comb_double_complex_rv] = fancy_timeit(comb_fun,num_reps);
ts_double['comb'] = display_time_stats(comb_double_complex_time,'Combined Complex Double')

#%%
'''
Sparse
'''
import scipy.sparse
import numpy as np
dtype = np.csingle
sm = 20000; sn = 20000
num_sp_el = int(1e6)
np.random.seed(1234)

def generate_random_sparse(shape,num_el,dtype):
    '''
    @brief fill a sparse array with numel_random elements
    '''
    data = (np.random.rand(num_el)+1j*np.random.rand(num_el)).astype(dtype)
    ri = np.random.randint(0,high=shape[0]-1,size=num_el)
    ci = np.random.randint(0,high=shape[1]-1,size=num_el)
    rv = scipy.sparse.coo_matrix((data,(ri,ci)),shape=shape,dtype=dtype)
    return rv  
    
sa = generate_random_sparse((sm,sn),num_sp_el,dtype)
sb = generate_random_sparse((sm,sn),num_sp_el,dtype)

sp_fun = lambda: sa.dot(sb)
[sp_double_complex_time,sp_double_complex_rv] = fancy_timeit(sp_fun,5);
ts_double['sparse'] = display_time_stats(sp_double_complex_time,'Combined Complex Double')

#%%
'''
Beamforming
'''
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumpy
#define our classes
mybf = SerialBeamformNumpy()
mybf.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float32, #floating point default to double
                'complexfloating':np.csingle, #defuault to complex128
                'integer':np.int32, #default to int32
                }
#define our values
freqs = [40e9] #frequency
freqs = np.arange(26.5e9,40e9,100e6,dtype=mybf.precision_types['floating'])
spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
numel = [10,10,1] #number of elements in x,y
Xel,Yel,Zel = np.meshgrid(np.arange(numel[0],dtype=mybf.precision_types['floating'])*spacing,
                          np.arange(numel[1],dtype=mybf.precision_types['floating'])*spacing,
                          np.arange(numel[2],dtype=mybf.precision_types['floating'])*spacing) #create our positions
pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
#az = np.arange(-90,90,1)
#el = np.arange(-90,90,1)
num_deg = 1000
az = np.random.rand(num_deg).astype(mybf.precision_types['floating'])
el = np.random.rand(num_deg).astype(mybf.precision_types['floating'])
inc_az = [0,45]; inc_el = [0,20]#np.zeros_like(inc_az)
#create incident waves
meas_vals = np.array([mybf.get_steering_vectors(freq,pos,inc_az,inc_el) for freq in freqs])
meas_vals = meas_vals.mean(axis=1)
weights = np.ones((pos.shape[0]),dtype=mybf.precision_types['complexfloating'])
#now create our lambda functions
num_freqs = len(freqs)
num_pos = pos.shape[0]
num_azel = len(az)
out_vals=np.zeros((num_freqs,num_azel),dtype=mybf.precision_types['complexfloating']);
bf_fun = lambda: (mybf._get_beamformed_values(freqs,pos,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel))
[bf_double_complex_time,bf_double_complex_rv] = fancy_timeit(bf_fun,10);
ts_double['div'] = display_time_stats(bf_double_complex_time,'Beamform Complex Double')

#%%
'''
write to file
'''
import json
with open(out_file_name,'w+') as fp:
    json.dump(ts_double,fp,indent=4, sort_keys=True)

#%%
'''
numba output
'''
ts_double = OrderedDict() #reset
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

#%%
'''
Add/Sub/Mult/Div
'''
add_fun = lambda : nb_add(a,b)
[add_double_complex_time,add_double_complex_rv] = fancy_timeit(add_fun,num_reps);
ts_double['add'] = display_time_stats(add_double_complex_time,'Add Complex Double')
sub_fun = lambda :  nb_subtract(a,b);
[sub_double_complex_time,sub_double_complex_rv] = fancy_timeit(sub_fun,num_reps);
ts_double['sub'] = display_time_stats(sub_double_complex_time,'Sub Complex Double')
mul_fun = lambda :  nb_multiply(a,b);
[mul_double_complex_time,mul_double_complex_rv] = fancy_timeit(mul_fun,num_reps);
ts_double['mul'] = display_time_stats(mul_double_complex_time,'Mul Complex Double')
div_fun = lambda :  nb_divide(a,b);
[div_double_complex_time,div_double_complex_rv] = fancy_timeit(div_fun,num_reps);
ts_double['div'] = display_time_stats(div_double_complex_time,'Div Complex Double')

#%%
'''
exp/matmul/sum
'''
#matmul_fun = lambda :  nb_matmul(a,b)
#[matmul_double_complex_time,matmul_double_complex_rv] = fancy_timeit(matmul_fun,num_reps);
ts_double['matmul'] = 'N/A'#= display_time_stats(matmul_double_complex_time,'Matrix Multiply Complex Double')

exp_fun = lambda :  nb_exp(a);
[exp_double_complex_time,exp_double_complex_rv] = fancy_timeit(exp_fun,num_reps);
ts_double['exp'] = display_time_stats(exp_double_complex_time,'Exponential Complex Double')

#sum_fun = lambda :  nb_sum(a);
#[sum_double_complex_time,sum_double_complex_rv] = fancy_timeit(sum_fun,num_reps);
ts_double['sum'] = 'N/A'
#= display_time_stats(sum_double_complex_time,'Sum Complex Double')

#%%
'''
LU Decomposition
'''
#lu_fun = lambda : lu(a)
#[lu_double_complex_time,lu_double_complex_rv] = fancy_timeit(lu_fun,num_reps);
ts_double['lu'] = 'N/A' #= display_time_stats(lu_double_complex_time,'LU Complex Double')

#%%
'''
FFT
'''
#np.random.seed(1234)
#fft_n = 2**23; #number of points in our fft (same as matlab bench)
#fft_data = np.random.rand(fft_n)+1j*np.random.rand(fft_n);
#fft_data = fft_data.astype(dtype)
#fft_fun = lambda : np.fft.fft(fft_data);
#[fft_double_complex_time,fft_double_complex_rv] = fancy_timeit(fft_fun,num_reps);
ts_double['fft'] = 'N/A' #= display_time_stats(fft_double_complex_time,'FFT Complex Double')

#%%
'''
Beamforming
'''
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumba
#define our classes
mybf = SerialBeamformNumba()
mybf.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float32, #floating point default to double
                'complexfloating':np.csingle, #defuault to complex128
                'integer':np.int32, #default to int32
                }
#define our values
freqs = [40e9] #frequency
freqs = np.arange(26.5e9,40e9,100e6,dtype=mybf.precision_types['floating'])
spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
numel = [10,10,1] #number of elements in x,y
Xel,Yel,Zel = np.meshgrid(np.arange(numel[0],dtype=mybf.precision_types['floating'])*spacing,
                          np.arange(numel[1],dtype=mybf.precision_types['floating'])*spacing,
                          np.arange(numel[2],dtype=mybf.precision_types['floating'])*spacing) #create our positions
pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
#az = np.arange(-90,90,1)
#el = np.arange(-90,90,1)
num_deg = 1000
az = np.random.rand(num_deg).astype(mybf.precision_types['floating'])
el = np.random.rand(num_deg).astype(mybf.precision_types['floating'])
inc_az = [0,45]; inc_el = [0,20]#np.zeros_like(inc_az)
#create incident waves
meas_vals = np.array([mybf.get_steering_vectors(freq,pos,inc_az,inc_el) for freq in freqs])
meas_vals = meas_vals.mean(axis=1)
weights = np.ones((pos.shape[0]),dtype=mybf.precision_types['complexfloating'])
#now create our lambda functions
num_freqs = len(freqs)
num_pos = pos.shape[0]
num_azel = len(az)
out_vals=np.zeros((num_freqs,num_azel),dtype=mybf.precision_types['complexfloating']);
bf_fun = lambda: (mybf._get_beamformed_values(freqs,pos,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel))
[bf_double_complex_time,bf_double_complex_rv] = fancy_timeit(bf_fun,10);
ts_double['beamform'] = display_time_stats(bf_double_complex_time,'Beamform Complex Double')

#%%
'''
combined operation
'''
comb_fun = lambda : nb_comb(a,b)
[comb_double_complex_time,comb_double_complex_rv] = fancy_timeit(comb_fun,num_reps);
ts_double['comb'] = display_time_stats(comb_double_complex_time,'Combined Complex Double')


'''
write to file
'''
import json
with open('nb_'+out_file_name,'w+') as fp:
    json.dump(ts_double,fp,indent=4, sort_keys=True)
    
def load_json(fname):
    with open(fname) as fp:
        return json.load(fp,object_hook=OrderedDict)
    
def print_speeds(fname,use_keys=True):
    key_order = ['add','sub','mul','div','matmul','exp','sum','lu','fft']
    vals = load_json(fname)
    for k in key_order:
        v = vals[k]
        if type(v) is OrderedDict:
            out_v = v['mean']
        else:
            out_v = v
        if use_keys:
            print("{}: {}".format(k,out_v))
        else:
            print(out_v)

