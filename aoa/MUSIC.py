# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 08:53:39 2019

@author: aweis
"""

from pycom.aoa.base.AoaAlgorithm import AoaAlgorithm,TestAoaAlgorithm
from numba import complex128,complex64
from numba import vectorize
import numpy as np
import cmath

class  Music(AoaAlgorithm):
    '''@brief a class for calculating aoa using MUSIC algorithm'''
    
    def calculate(self,freqs,pos,meas_vals,az,el,**kwargs):
        '''
        @brief calculate music aoa estimate
        @param[in] freqs - frequencies to calcualte for
        @param[in] pos - list of [x,y,z] position values 
        @param[in] meas_vals - measured complex values at each element. 
                should be size (len(freqs),len(pos))
        @param[in] az - list of azimuth angles in radians
        @param[in] el - list of elevation angles in radians
        @param[in/OPT] kwargs - keyword args as follows:
            - number_of_signals - specify the number of signals for music.
                Otherwise use default
            - subarray_idx - indexes to subarray our positions to give the required 
                number of samples for music. This will auto split if not provided
        '''
        #first we need to generate our default values
        options = {}
        options['number_of_signals'] = 1
        options['subarray_idx'] = None
        for k,v in kwargs.items():
            options[k] = v
        #generate our steering vectors
        sv = self.get_steering_vectors(freqs,pos,meas_vals,az,el)
        num_sigs = options['number_of_signals']; #number of signals we expect to see

        #now lets calculate music
        for fn in range(len(freqs)): #loop through all freqs
            fmv = meas_vals[fn] #get meas vals at the current frequency
            mcov = np.matmul(np.array(fmv).reshape(-1,1),np.array(fmv).reshape(1,-1))/(len(fmv)-1)
            [lhat,uhat] = scipy.linalg.eig(mcov); #get our eigenvectors from the cov
            order = np.argsort(np.diag(lhat))[::-1]; #sort in ascending order
            sort_uhat = uhat[:,order]; #sort the vectors too
    
            s_fun = lambda azv,elv: np.exp(1j*(positions*get_k_vector_azel(freqs(fn),azv,elv)));
            ms_vals = zeros(1,length(az));
            for an in range(len(az)):
                s = s_fun(az(an),el(an));
                s = split_by_col_idx(s,options['subarray_idx']);
                ms_vals[i] = 1./np.sum(np.sum(np.abs(s.T*sort_uhat[:,1:end-num_sigs])))**2;
            out_vals[fn,:] = ms_vals;  