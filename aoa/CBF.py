# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 08:52:40 2019

@author: aweis
"""

from pycom.aoa.base.AoaAlgorithm import AoaAlgorithm

class CBF(AoaAlgorithm):
    '''@brief perform conventional beamforming'''
    
    @classmethod
    def calculate(self,freqs,pos,meas_vals,az,el,**kwargs):
        '''
        @brief calculate and return values o the algorithm for the given input
        @param[in] freqs - np array of frequencies to calculate at
        @param[in] pos - xyz positions to perform at
        @param[in] meas_vals - measured values at each position and frequency of shape (len(freqs),len(pos))
        @param[in] az - np.array of azimuthal angles to calculate radians
        @param[in] el - np.array of elevations radians
        @param[in] kwargs - keyword args as follows:
            weights - list of complex weightings to add to each position (taperings)
        '''
        raise NotImplementedError