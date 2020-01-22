# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 08:52:40 2019

@author: aweis
"""

from pycom.aoa.base.AoaAlgorithm import AoaAlgorithm,TestAoaAlgorithm
from pycom.aoa.base.AoaAlgorithm import vector_mult3_complex
from numba import complex128,complex64
from numba import vectorize, njit
import numpy as np
from numpy import sum as nsum
from cmath import exp as cexp

@vectorize([complex64 (complex64 ,complex64 ,complex64 ),
            complex128(complex128,complex128,complex128)],target='parallel')
def vector_beamform(weights,meas_vals,sv_no_exp):
    return weights*meas_vals*cexp(-1j*sv_no_exp)

class CBF(AoaAlgorithm):
    '''@brief perform conventional beamforming'''
    
    @classmethod
    def calculate(cls,freqs,pos,meas_vals,az,el,**kwargs):
        '''
        @brief calculate and return values o the algorithm for the given input  
        @param[in] freqs - np array of frequencies to calculate at  
        @param[in] pos - xyz positions to perform at  
        @param[in] meas_vals - measured values at each position and frequency of shape (len(freqs),len(pos))  
        @param[in] az - np.array or single value of azimuthal angles to calculate radians  
        @param[in] el - np.array or single value of elevations radians  
        @param[in] kwargs - keyword args as follows:  
            - weights - list of complex weightings to add to each position (taperings) 
            - verbose - be verbose in calculations
        @return complex beamformed value for each frequency and angle with dtype=meas_vals.dtype
        '''
        if np.ndim(freqs)<1: freqs = np.asarray([freqs])
        if np.ndim(el   )<1: el    = np.asarray([el   ])
        if np.ndim(az   )<1: az    = np.asarray([az   ])
        if np.ndim(pos)<1:   pos = np.asarray([pos])
        
        if 'weights' not in kwargs:
            kwargs['weights'] = np.ones((len(pos),),dtype=meas_vals.dtype)
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
            
        out_vals = np.ndarray((freqs.size,az.size),dtype=meas_vals.dtype)
        weights = kwargs['weights']
        weight_sum = nsum(weights)
        for fn in range(len(freqs)):
            if kwargs['verbose']: print("Running with frequency {}".format(freqs[fn]))
            sv = np.squeeze(cls.get_steering_vectors_no_exp(freqs[fn],pos,az,el,dtype=meas_vals.dtype))
            temp_mult = vector_beamform(weights[:],meas_vals[fn,:],sv[:])
            out_vals[fn,:] = nsum(temp_mult,axis=-1)/weight_sum #divide by weights
            
        return out_vals
            
    @classmethod
    def get_steering_vectors_no_exp(cls,freqs,pos,az,el,**kwargs):
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
        kvecs = cls.get_k_vector_azel(freqs,az,el)
        steering_vecs_out = np.ndarray((len(freqs),len(az),len(pos)),dtype=options['dtype'])
        for fn in range(len(freqs)):
            steering_vecs_out[fn,...] = np.matmul(kvecs[fn,...],pos.T)
        return steering_vecs_out
            
            
class TestCBF(TestAoaAlgorithm):
    '''@brief unittest class for aoa algorithm'''
    def set_options(self):
        self.options['aoa_class'] = CBF
            
if __name__=='__main__':
    
    from WeissTools.python.PlotTools import format_plot
    
    mytest = TestCBF()
    odf = mytest.plot_1d_calc()
    tdf = mytest.plot_2d_calc()
    odf = format_plot(odf)
    tdf = format_plot(tdf,font_size=12)
    
    odf.write_html('../docs/aoa/cbf/figs/1D_results.html')
    tdf.write_html('../docs/aoa/cbf/figs/2D_results.html')
   # save_plot(tdf,'beamforming_results','../docs/aoa/cbf')
#    mycbf = CBF()
#    az,el = mytest.angles_1d
#    pos = mytest.positions_1d
#    freqs = mytest.freqs
#    meas_vals = mytest.meas_vals_1d
#    out_vals = mycbf.calculate(freqs,pos,meas_vals,az,el)
#    import matplotlib.pyplot as plt
#    plt.plot(az,np.abs(out_vals.T))