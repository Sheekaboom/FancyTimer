# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:14:54 2019

@author: aweiss
"""
import os
os.environ['NUMBA_NUM_THREADS'] = '12' # match the number used in MATLAB
from numba import config as nb_config
print(nb_config.NUMBA_NUM_THREADS)

from pycom.timing.OperationTimer import fancy_timeit_matrix_sweep
from pycom.timing.beamforming.beamform_speed import beamform_speed
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
    dim_list = np.floor(np.linspace(1,50,6)).astype(np.uint32);
    num_reps = 10
else:
    dim_list = np.floor(np.linspace(1,500,51)).astype(np.uint32);
    num_reps = 100
    
out_file = os.path.join(output_directory,'stats_py_np_beamforming.mat')


stats_np_single = {}
stats_np_double = {}

#%% argument generation
def beamforming_arg_gen_funct(dim,num_args): #generate num_cells_x,num_cells_y as dim
    return [dim,np.cdouble]  
def beamforming_arg_gen_funct_single(dim,num_args): #generate num_cells_x,num_cells_y as dim
    return [dim,np.csingle]  

#%% now run
[solve_stats_double,_] = fancy_timeit_matrix_sweep(
        [beamform_speed],['beamforming'],[2],dim_list,num_reps
        ,arg_gen_funct=beamforming_arg_gen_funct);
[solve_stats_single,_] = fancy_timeit_matrix_sweep(
        [beamform_speed],['beamforming'],[2],dim_list,num_reps
        ,arg_gen_funct=beamforming_arg_gen_funct_single);
		
stats_np_double.update(solve_stats_double)
stats_np_single.update(solve_stats_single)
    
#%% now save out (single is not used)
import scipy.io as sio
sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})
    