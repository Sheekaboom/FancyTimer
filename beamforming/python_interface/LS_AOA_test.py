# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:20:04 2019
This is a test of a least squares aoa estimator. This is going to work as follows:
    1. prune angles (A_p):
        take measured data and set a lower bound for where AoAs may be. This will reduce the 
        number of angles that need to be calculated
    2. create the basis set (basis):
        beamform at all angles in measured angles for a single incident plane wave
        for each angle in our pruned angles. This will be our basis vectors
    3. Solve the least squares problem:
        now we just have to fit our original data to our basis set using a least squares approximation
        
This code is going to test the above algorithm for the 1D case at a single frequency

@author: aweis
"""
import numpy as np
import matplotlib.pyplot as plt
from SerialBeamform import SerialBeamformNumba

#%% initialize some things
mybf = SerialBeamformNumba()
#freq to calculate at
freq = 40e9
#angles for AoA
az = np.arange(-90,91)
el = np.zeros_like(az)

#%% create our array. Use XYZ to adhere to 3D codes
spacing = 2.99e8/np.max(freq)/2 #get our lambda/2
numel = [10,1,1] #number of elements in x,y
Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list

#%% now lets create some synthetic data
num_incident_waves = 1 #the number of incident plane waves to generate
mag_range_db = [-80,-40] #range of the magnitudes in db [min,max] #TODO implement
#mag_range_
np.random.seed(1234)
inc_az = (0.5-np.random.rand(num_incident_waves))*90 # restrict from -90 to 90
inc_az = np.array([27])
inc_el = np.zeros_like(inc_az)
meas_vals = np.sum(mybf.get_steering_vectors(freq,pos,inc_az,inc_el),axis=0)

#now beamform our values to fit with our basis
bf_vals = mybf.get_beamformed_values([freq],pos,np.ones_like(meas_vals),[meas_vals],az,el)
#and plot for testing
plt.plot(az,10*np.log10(np.abs(bf_vals.squeeze())))

#%% Now generate our basis set
base_angles = np.arange(-60,61) #assume we are only looking from -60 to 60 for now
base_angles = np.arange(-90,91)
basis = np.ndarray((len(base_angles),len(az)),dtype=meas_vals.dtype)
for i,ang in enumerate(base_angles): #for each azimuth create an incident plane wave and beamform
    #this could be pruned in the future to reduce runtime for 2D case
    cur_inc = mybf.get_steering_vectors(freq,pos,np.array([ang]),np.array([0]))[0]
    basis[i,:] = mybf.get_beamformed_values([freq],pos,np.ones_like(cur_inc),[cur_inc],az,el)
    
plt.plot(az,10*np.log10(np.abs(basis.transpose())))

#%% now calculate our least squares fit
np.lstsq(basis)



