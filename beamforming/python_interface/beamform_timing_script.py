# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:15:59 2019
Beamform timing script
@author: aweis
"""

import timeit
import numpy as np
from SerialBeamform import SerialBeamformFortran,SerialBeamformNumpy,SerialBeamformNumba,SerialBeamformPython
n = 2
myfbf = SerialBeamformFortran()
mypbf = SerialBeamformPython()
mynpbf = SerialBeamformNumpy()
mynbbf = SerialBeamformNumba()
freqs = [40e9] #frequency
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
print('FORTRAN: ',end='')
ft = timeit.Timer(lambda: myfbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
print(ft.timeit(number=5)/5.)
print('NUMPY: ',end='')
npt = timeit.Timer(lambda: mynpbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
print(npt.timeit(number=5)/5.)
print('NUMBA: ',end='')
nbt = timeit.Timer(lambda: mynbbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
print(npt.timeit(number=5)/5.)
print('PYTHON: ',end='')
print("DISABLED")
#pt = timeit.Timer(lambda: mypbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el))
#print(pt.timeit(number=1))/1

