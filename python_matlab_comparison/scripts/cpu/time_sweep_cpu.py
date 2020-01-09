# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 20:09:23 2019

@author: aweiss
"""
#run each of these scripts by importing them

DEBUG=False
output_directory = './data'

print("Running Basic Sweep")
import time_sweep_basic_cpu
print("Running Extended Sweep")
import time_sweep_extended_cpu
print("Running Matrix Sweep")
import time_sweep_matrix_cpu
print("Running Sparse Sweep")
import time_sweep_sparse_cpu
print("Running FDFD Sweep")
import time_sweep_fdfd_cpu
print("Running Beamforming Sweep")
import time_sweep_beamforming_cpu