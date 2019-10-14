'''
@author ajw
@date 8-24-2019
@brief functions and classes for creating
    python functions from fortran subroutines
'''
import numpy as np
import os

from SpeedBeamform import SpeedBeamform,PythonBeamform
from SpeedBeamform import get_ctypes_pointers
import ctypes
wdir = os.path.dirname(os.path.realpath(__file__))

############################################
### FORTRAN implementation
############################################
class SerialBeamformFortran(SpeedBeamform):
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    serial_fortran_library_path = os.path.join(wdir,'../fortran/build/libbeamform_serial.dll')
    def __init__(self):
        '''
        @brief constructor
        '''
        super().__init__(self.serial_fortran_library_path)
        self.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float64, #floating point default to double
                'complexfloating':np.cdouble, #defuault to complex128
                'integer':np.int32, #default to int32
                }

############################################
### PYTHON implementation
############################################
import math
import cmath
class SerialBeamformPython(PythonBeamform):
    '''
    @brief mostly purely python beamforming implementation (try to use math and not numpy)
    @note some things still require numpy for things like np.exp
    '''
    def __init__(self):
        '''
        @brief constructor
        @note a class is defined here to set values of self._lib
        '''
        super().__init__()
        #define 

    def _get_steering_vectors(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        '''
        @brief override to utilize for python engine
        '''
        
        for i in range(num_azel):
            kvec = self._get_k_vector_azel(freq,az[i],el[i])
            for p in range(num_pos):
                steering_vecs_out[i,p] = cmath.exp(sum([-1j*kvec[pxyz]*positions[p,pxyz] for pxyz in range(3)]))
                
    def _get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel):
        '''
        @brief override to utilize for a python engine
        '''
        num_pos = positions.shape[0]
        num_azel = az.shape[0]
        sv = np.zeros((num_azel,num_pos),dtype=self.precision_types['complexfloating'])
        for fn in range(num_freqs):
            #print("Running with frequency {}".format(freqs[fn]))
            self._get_steering_vectors(freqs[fn],positions,az,el,sv,num_pos,num_azel)
            for an in range(num_azel):
                out_vals[fn,an] = sum([weights[p]*meas_vals[fn,p]*sv[an,p] for p in range(num_pos)])/num_pos

############################################
### NUMPY implementation
############################################
class SerialBeamformNumpy(PythonBeamform):
    '''
    @brief python beamforming heavily relying on numpy
    @note some things still require numpy for things like np.exp
    '''
    def __init__(self):
        '''
        @brief constructor
        @note a class is defined here to set values of self._lib
        '''
        super().__init__()
        #define

    def _get_steering_vectors(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        '''
        @brief override to utilize for python engine
        '''       
        kvec = self._get_k_vector_azel(freq,az,el)
        steering_vecs_out[:,:] = np.exp(-1j*np.matmul(positions,kvec.transpose())).transpose()
                
    def _get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel):
        '''
        @brief override to utilize for a python engine
        '''
        num_pos = positions.shape[0]
        num_azel = az.shape[0]
        sv = np.zeros((num_azel,num_pos),dtype=self.precision_types['complexfloating'])
        for fn in range(num_freqs):
            #print("Running with frequency {}".format(freqs[fn]))
            self._get_steering_vectors(freqs[fn],positions,az,el,sv,num_pos,num_azel)
            out_vals[fn,:] = np.sum(weights*meas_vals[fn]*sv,axis=-1)/num_pos
 
from numba import vectorize, complex64,float32
import cmath           
############################################
### Numba vector implementation 
############################################
class SerialBeamformNumba(PythonBeamform):
    '''
    @brief python beamforming mixing numba vectorize and numpy
    @note some things still require numpy for things like np.exp
    '''
    def __init__(self):
        '''
        @brief constructor
        @note a class is defined here to set values of self._lib
        '''
        super().__init__()
        #define vector operation inputs

    def _get_steering_vectors(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        '''
        @brief override to utilize for python engine
        '''       
        kvec = self._get_k_vector_azel(freq,az,el)
        for an in range(num_azel):
            steering_vecs_out[an,:] = self.vector_exp_complex(np.sum(self.vector_mult_complex(positions,kvec[an]),axis=-1))
    
    def _get_steering_vectors_no_exp(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        kvec = self._get_k_vector_azel(freq,az,el)
        for an in range(num_azel):
            steering_vecs_out[an,:] = (np.sum(self.vector_mult_complex(positions,kvec[an]),axis=-1))

    def _get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel):
        '''
        @brief override to utilize for a python engine
        '''
        num_pos = positions.shape[0]
        num_azel = az.shape[0]
        sv = np.zeros((num_azel,num_pos),dtype=self.precision_types['complexfloating'])
        for fn in range(num_freqs):
            #print("Running with frequency {}".format(freqs[fn]))
            self._get_steering_vectors_no_exp(freqs[fn],positions,az,el,sv,num_pos,num_azel)
            temp_mult = self.vector_beamform(weights,meas_vals,sv)
            out_vals[fn,:] = np.sum(temp_mult,axis=-1)/num_pos
            
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
    

if __name__=='__main__':
    from pycom.beamforming.python_interface.SpeedBeamform import SpeedBeamformUnittest
    import unittest
    
    class NumpyUnittest(SpeedBeamformUnittest,unittest.TestCase):
        def set_beamforming_class(self):
            self.beamforming_class = SerialBeamformNumpy()
            
    class NumbaUnittest(SpeedBeamformUnittest,unittest.TestCase):
        def set_beamforming_class(self):
            self.beamforming_class = SerialBeamformNumba()
    
    class FortranUnittest(SpeedBeamformUnittest,unittest.TestCase):
        def set_beamforming_class(self):
            self.beamforming_class = SerialBeamformFortran()
    
    test_class_list = [NumpyUnittest,NumbaUnittest,FortranUnittest]
    tl = unittest.TestLoader()
    suite = unittest.TestSuite([tl.loadTestsFromTestCase(mycls) for mycls in test_class_list])
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    
    from collections import OrderedDict
    n = 2
    beamformer_classes = OrderedDict()
    beamformer_classes['NUMPY']    = SerialBeamformNumpy
    beamformer_classes['NUMBA']    = SerialBeamformNumba
    beamformer_classes['FORTRAN']  = SerialBeamformFortran
    #beamformer_classes['Python']   = SerialBeamformPython
    beamformers = {k:v() for k,v in beamformer_classes.items()}
    
    bf_base = 'NUMPY' #class to compare everything against

    #test get steering vectors
    freqs = [40e9] #frequency
    #freqs = np.arange(26.5e9,40e9,10e6)
    spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
    numel = [35,35,1] #number of elements in x,y
    #numel = [5,1,1]
    Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
    pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
    az = np.arange(-90,90,1)
    el = np.arange(-90,90,1)
    np.set_printoptions(precision=16,floatmode='fixed')
    print("STEERING VECTOR EQUALITY CHECK:")
    base_sv = beamformers[bf_base].get_steering_vectors(freqs[0],pos,az,el) #our base values to compare to
    rel_err = 1e-12 #relative error allowed between elements (to account for machine error between python/c/fortran)
    for k,v in beamformers.items():
        cur_sv = v.get_steering_vectors(freqs[0],pos,az,el)
        sv_eq = np.isclose(base_sv,cur_sv,rtol=rel_err,atol=0)
        print("{}: {}".format(k,np.all(sv_eq)))
        if not np.all(sv_eq): #print the offenders and the difference
            sv_neq = np.invert(sv_eq) #where they arent equal
            err = ((base_sv[sv_neq]-cur_sv[sv_neq])/base_sv[sv_neq])
            _,neq_idx = np.where(sv_neq) #how many places they arent equal
            print("    # Not Equal Elements:   {}".format(len(neq_idx)))
            print("    Max Error           :   {}".format(np.abs(np.max(err))))
            print("    Max Error Element   :   {}".format(np.argmax(err)))
            bnsv = base_sv[sv_neq]
            cnsv = cur_sv[sv_neq]
    
    #create synthetic incident plane wave data
    inc_az = [0,45]; inc_el = [0,20]#np.zeros_like(inc_az)
    meas_vals = np.array([beamformers[bf_base].get_steering_vectors(freq,pos,inc_az,inc_el) for freq in freqs])
    meas_vals = meas_vals.mean(axis=1)

    weights = np.ones((pos.shape[0]),dtype=np.complex128)
    bf_vals = OrderedDict()
    for k,v in beamformers.items():
        vals = v.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
        bf_vals[k] = vals
    '''
    #and plot
    import matplotlib.pyplot as plt
    freq_to_plot = 0
    for k,v in bf_vals.items():
        plt_vals = 10*np.log10(np.abs(v[freq_to_plot]))
        plt.plot(az,plt_vals,label=k)
    plt.legend()
    '''
    '''
    #test array multiply
    arr1 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arr2 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arro = np.zeros_like(arr1)
    mysbf.test_complex_array_multiply(arr1.ctypes,arr2.ctypes,arro.ctypes,cnp)
    nparro = arr1*arr2
    print(all(nparro==arro))
    
    #test matrix multiply
    mat1 = (np.random.rand(n,n)+1j*np.random.rand(n,n)).astype(np.csingle)
    mat2 = (np.random.rand(n,n)+1j*np.random.rand(n,n)).astype(np.csingle)
    mato = np.zeros_like(mat1)
    mysbf.test_complex_matrix_multiply(mat1.ctypes,mat2.ctypes,mato.ctypes,cnp)
    npmato = mat1*mat2
    
    #test 2d sum
    sarro = np.zeros_like(arr1)
    mysbf.test_complex_sum(mat1.ctypes,sarro.ctypes,cnp)
    npsarro = np.sum(mat1,axis=0)
    print(all(npsarro==sarro))
    '''    
        
        
        
        
        
    
    
    