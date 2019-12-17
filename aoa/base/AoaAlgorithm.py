'''
@brief some base classes and functions for AoA estimation algorithms
This will have relatively fast implementation of some things (like steering vectors)
@author Alec Weiss
@date  11/2019
'''

import numpy as np
import cmath
from numba import vectorize

class AoaAlgorithm:
    '''@brief this is a base class for creating aoa algorithms'''
    SPEED_OF_LIGHT = np.double(299792458.0)
    def __init__(*args,**kwargs):
        '''@brief constructor. Dont do anything right now'''
        pass
    
    #%% Lots of methods that should be defined by subclasses
    @classmethod
    def get_k(self,freq,eps_r=1,mu_r=1):
        '''
        @brief get our wavenumber
        @note this part stays the same for all python implementations
        @example
        >>> AoaAlgorithm.get_k(40e9,1,1)
        838.3380087806727
        '''
        lam = self.SPEED_OF_LIGHT/np.sqrt(eps_r*mu_r)/freq
        k = 2*np.pi/lam
        return k 
    
    @classmethod
    def get_k_vector_azel(self,freq,az,el,**kwargs):
        '''
        @brief get our k vectors (e.g. kv_x = k*sin(az)*cos(el))
        @param[in] freq - frequency in hz
        @param[in] az - azimuth in radians (0 is boresight)
        @param[in] el - elevation in radians (0 boresight)
        @param[in] kwargs - keyword args passed to get_k
        @example
        >>> import numpy as np
        >>> AoaAlgorithm.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        array([[592.7944909352411287,   0.0000000000000000, 592.7944909352411287],
               [642.2041730818633596,   0.0000000000000000, 538.8732847735016094]])
        '''
        k = self.get_k(freq,**kwargs)
        k = k.reshape(-1,1,1)
        kvec = np.array([
                np.sin(az)*np.cos(el),
                np.sin(el),
                np.cos(az)*np.cos(el)]).transpose()
        kvec = k*kvec[np.newaxis,...]
        return kvec
    
    @classmethod
    def get_steering_vectors(self,freqs,pos,az,el,**kwargs):
        '''
        @brief return a set of steering vectors for a list of az/el pairs
        @param[in] freqs - frequencies to calculate for
        @param[in] pos - list of xyz positions of points in the array
        @param[in] az - np.array of azimuthal angles in radians
        @param[in] el - np.array of elevations in radians
        @return np.ndarray of size (len(freqs),len(az),len(pos))
        '''
        if len(freqs)==1: freqs = [freqs] #ensure we have a list
        freqs = np.array(freqs) #change to nparray
        kvecs = self.get_k_vector_azel(freqs,az,el)
        steering_vecs_out = np.ndarray((len(freqs),len(az),len(pos)),dtype=np.cdouble)
        for fn in range(len(freqs)):
            steering_vecs_out[fn,...] = self.vector_exp_complex(np.matmul(pos,kvecs[fn,...].transpose())).transpose()
        return steering_vecs_out
    
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
    
    #%% numba vectorize operations
    from numba import complex128,complex64
    @vectorize(['complex128(complex128)'],target='parallel')
    def vector_exp_complex(vals):
        return cmath.exp(-1j*vals)
    
    @vectorize(['complex128(complex128,complex128)'],target='parallel')
    def vector_mult_complex(a,b):
        return a*b

    @vectorize([complex128(complex128,complex128,complex128),
                complex64 (complex64 ,complex64 ,complex64 )],target='cpu')
    def vector_beamform(weights,meas_vals,sv_no_exp):
        return weights*meas_vals*cmath.exp(-1j*sv_no_exp)

#%% unit testing
import unittest

class TestAoaAlgorithm(unittest.TestCase):
    '''@brief basic unittesting of AoaAlgorithm object'''
    AOA_CLASS = AoaAlgorithm #possibillity for testing multiple classes
    def test_get_k(self):
        '''@brief test getting the wavenumber'''
        myaoa = self.AOA_CLASS #test both static and object
        o_ans = myaoa.get_k(40e9,1,1);
        s_ans = AoaAlgorithm.get_k(40e9,1,1);
        expected_ans = 838.3380087806727
        self.assertEqual(o_ans,expected_ans)
        self.assertEqual(s_ans,expected_ans)
        
    def test_get_k_vector_azel(self):
        '''@brief test getting k vector in azel'''
        myaoa = self.AOA_CLASS()
        o_ans = myaoa.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        s_ans = AoaAlgorithm.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        expected_ans = np.array([[592.7944909352411,   0.000, 592.7944909352411],
                                 [642.2041730818634,   0.000, 538.8732847735016]])
        self.assertTrue(np.all(np.squeeze(o_ans)==expected_ans))
        self.assertTrue(np.all(np.squeeze(s_ans)==expected_ans))
            
#%% main tests
if __name__=='__main__':
    unittest.main()
    mya = AoaAlgorithm()
    freqs = np.arange(1,25)
    px = np.arange(-3,4); py = np.zeros_like(px); pz = np.zeros_like(px)
    pos = np.array([px,py,pz]).transpose()
    az = np.linspace(-np.pi/2,np.pi/2,10)
    el = np.zeros_like(az)
    kv = mya.get_k_vector_azel(freqs,az,el)
    sv = mya.get_steering_vectors(freqs,pos,az,el)
#    suite = unittest.TestSuite([TestAoaAlgorithm])
#    suite.run()
    
        
    


