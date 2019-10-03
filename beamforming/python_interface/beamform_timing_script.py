# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:15:59 2019
Beamform timing script
@author: aweis
"""

import timeit
import numpy as np
from SerialBeamform import SerialBeamformFortran,SerialBeamformNumpy,SerialBeamformNumba,SerialBeamformPython
from MatlabBeamform import SerialBeamformMatlab
n = 2
myfbf = SerialBeamformFortran()
mypbf = SerialBeamformPython()
mynpbf = SerialBeamformNumpy()
mynbbf = SerialBeamformNumba()
mymbf = SerialBeamformMatlab()
freqs = [40e9] #frequency
#freqs = np.arange(26.5e9,30.1e9,0.5e9)
spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
numel = [35,35,1] #number of elements in x,y
Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
az = [-45]
el = np.zeros_like(az)
sv = myfbf.get_steering_vectors(freqs[0],pos,az,el)
psv = mypbf.get_steering_vectors(freqs[0],pos,az,el)
npsv = mynpbf.get_steering_vectors(freqs[0],pos,az,el)
meas_vals = np.tile(sv[0],(len(freqs),1)) #syntethic plane wave
weights = np.ones((pos.shape[0]),dtype=np.complex128)
azl = np.arange(-90,90,1)
ell = np.arange(-90,90,1)
AZ,EL = np.meshgrid(azl,ell)
az = AZ.flatten()
el = EL.flatten()

    
print('TIMING:')
import timeit
num_reps = 3
#run fortran
print("Timing FORTRAN")
ft = timeit.Timer(lambda: myfbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
ftt = ft.timeit(number=num_reps)
#run numpy
print("Timing NUMPY")
npt = timeit.Timer(lambda: mynpbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
nptt = npt.timeit(number=num_reps)
#run numba
print("Timing NUMBA")
nbt = timeit.Timer(lambda: mynbbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
nbtt = nbt.timeit(number=num_reps)
#run Matlab
print("Timing MATLAB")
mbt = timeit.Timer(lambda: mymbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
mbtt = nbt.timeit(number=num_reps)
#run python
'''
print("Timing PYTHON")
pt = timeit.Timer(lambda: mypbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
#ptt = pt.timeit(number=num_reps)
ptt = np.nan
'''

def print_statistics(run_time,num_reps,baseline_time):
    '''
    @brief print statistics on timing
    @param[in] run_time - return from timeit.Timer.timeit()
    @param[in] num_reps - number of repeats when timing
    @param[in] baseline_time - time to compare against for speedup
    '''
    print("    Total Run Time: {}".format(run_time))
    print("    Single Run Time: {}".format(run_time/float(num_reps)))
    print("    Speedup over baseline: {}".format(baseline_time/run_time))

print('STATISTICS')
#print('PYTHON: ')
#print_statistics(ptt,num_reps,nptt)
print('NUMPY: ')
print_statistics(nptt,num_reps,nptt)
print('NUMBA: ')
print_statistics(nbtt,num_reps,nptt)
print('MATLAB: ')
print_statistics(mbtt,num_reps,nptt)
print('FORTRAN: ')
print_statistics(ftt,num_reps,nptt)



