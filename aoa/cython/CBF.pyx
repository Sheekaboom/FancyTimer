# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:19:16 2020
Implementation of conventional beamforming in Cython
@author: aweis
"""

from pycom.aoa.base.AoaAlgorithm import AoaAlgorithm,TestAoaAlgorithm
from numba import complex128,complex64
from numba import vectorize
import numpy as np
import cmath

class CBF(AoaAlgorithm):
    '''@brief perform conventional beamforming'''
    
    @classmethod
    def calculate(self,freqs,pos,meas_vals,az,el,**kwargs):
        '''
        @brief calculate and return values o the algorithm for the given input  
        @param[in] freqs - np array of frequencies to calculate at  
        @param[in] pos - xyz positions to perform at  
        @param[in] meas_vals - measured values at each position and frequency of shape (len(freqs),len(pos))  
        @param[in] az - np.array or single value of azimuthal angles to calculate radians  
        @param[in] el - np.array or single value of elevations radians  
        @param[in] kwargs - keyword args as follows:  
            - weights - list of complex weightings to add to each position (taperings) 
        @return complex beamformed value for each frequency and angle with dtype=meas_vals.dtype
        '''
        if np.ndim(freqs)<1: freqs = np.asarray([freqs])
        if np.ndim(el   )<1: el    = np.asarray([el   ])
        if np.ndim(az   )<1: az    = np.asarray([az   ])
        if 'weights' not in kwargs:
            kwargs['weights'] = np.ones((len(pos),),dtype=meas_vals.dtype)
        out_vals = np.ndarray((freqs.size,az.size),dtype=meas_vals.dtype)
        for fn in range(len(freqs)):
            #print("Running with frequency {}".format(freqs[fn]))
            sv = self.get_steering_vectors_no_exp(freqs[fn],pos,az,el,dtype=meas_vals.dtype)
            temp_mult = self.vector_beamform(kwargs['weights'],meas_vals[fn],sv)
            out_vals[fn,:] = np.sum(temp_mult,axis=-1)/np.sum(kwargs['weights']) #divide by weights
        return out_vals
            
    def vector_beamform(weights,meas_vals,sv_no_exp):
        return weights*meas_vals*np.exp(-1j*sv_no_exp)
    
    @classmethod
    def get_steering_vectors_no_exp(self,freqs,pos,az,el,**kwargs):
        '''
        @brief return a set of steering vectors for a list of az/el pairs without doing complex exponential  
        @param[in] freqs - frequencies to calculate for    
        @param[in] pos - list of xyz positions of points in the array    
        @param[in] az - np.array of azimuthal angles in radians  
        @param[in] el - np.array of elevations in radians  
        @param[in/OPT] kwargs - keyword args as follows:
            - dtype - complex data type for output (default np.cdouble)
        @return np.ndarray of size (len(freqs),len(az),len(pos))  
        '''  
        if np.ndim(freqs)<1: freqs = np.asarray([freqs])
        if np.ndim(az)<1:    az = np.asarray([az])
        if np.ndim(el)<1:    el = np.asarray([el])
        if np.ndim(pos)<1:   pos = np.asarray([pos])
        options = {}
        options['dtype'] = np.cdouble
        for k,v in kwargs.items():
            options[k] = v
        freqs = np.asarray(freqs) #change to nparray
        kvecs = self.get_k_vector_azel(freqs,az,el)
        steering_vecs_out = np.ndarray((len(freqs),len(az),len(pos)),dtype=options['dtype'])
        for fn in range(len(freqs)):
            steering_vecs_out[fn,...] = np.matmul(pos,kvecs[fn,...].transpose()).transpose()
        return steering_vecs_out
            
            
class TestCBF(TestAoaAlgorithm):
    '''@brief unittest class for aoa algorithm'''
    def set_options(self):
        self.options['aoa_class'] = CBF
