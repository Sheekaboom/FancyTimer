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
            out_vals[:,:] = np.sum(weights*meas_vals[fn]*sv,axis=-1)/num_pos
 
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
            temp_mult = self.vector_mult_complex(weights,meas_vals[fn])
            temp_mult = self.vector_mult_complex(temp_mult,sv)
            out_vals[:,:] = np.sum(temp_mult,axis=-1)/num_pos
            
    
    @vectorize(['complex128(complex128)'],target='cpu')
    def vector_exp_complex(vals):
        return cmath.exp(-1j*vals)
    
    @vectorize(['complex128(complex128,complex128)'],target='cpu')
    def vector_mult_complex(a,b):
        return a*b
    

if __name__=='__main__':
    n = 2
    myfbf = SerialBeamformFortran()
    mypbf = SerialBeamformPython()
    mynpbf = SerialBeamformNumpy()
    mynbbf = SerialBeamformNumba()
    
    #test get steering vectors
    freqs = [40e9] #frequency
    #freqs = np.arange(26.5e9,40e9,10e6)
    spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
    numel = [35,1,1] #number of elements in x,y
    Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
    pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
    #az = np.arange(-90,90,1)
    #az = np.array([-90,45,0,45 ,90])
    az = [-45]
    el = np.zeros_like(az)
    sv = myfbf.get_steering_vectors(freqs[0],pos,az,el)
    psv = mypbf.get_steering_vectors(freqs[0],pos,az,el)
    npsv = mynpbf.get_steering_vectors(freqs[0],pos,az,el)
    nbsv = mynbbf.get_steering_vectors(freqs[0],pos,az,el)
    print("STEERING VECTOR EQUALITY CHECK (4 decimal places):")
    rdp = 2
    print(np.all(np.round(sv,rdp)==np.round(psv,rdp)))
    print(np.all(np.round(npsv,rdp)==np.round(psv,rdp)))
    print(np.all(np.round(nbsv,rdp)==np.round(psv,rdp)))
    meas_vals = np.tile(sv[0],(len(freqs),1)) #syntethic plane wave
    #meas_vals = np.ones((1,35))
    #print(np.rad2deg(np.angle(sv)))
    #print(sv)
    #print(np.real(sv))
    azl = np.arange(-90,90,1)
    ell = np.arange(-90,90,1)
    AZ,EL = np.meshgrid(azl,ell)
    az2 = AZ.flatten()
    el2 = EL.flatten()
    #az = np.array([-90,45,0,45,90])
    #az = [-90]
    az = azl
    el = np.zeros_like(az)
    weights = np.ones((pos.shape[0]),dtype=np.complex128)
    bf_vals = myfbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
    pbf_vals = mypbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
    npbf_vals = mynpbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
    nbbf_vals = mynbbf.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
    #print(bf_vals)
    #print(get_k_vec(freq,az,el))
        
        
    import matplotlib.pyplot as plt
    freq_to_plot = 0
    plt.plot(az,10*np.log10(np.abs(bf_vals[freq_to_plot])),label='FORTRAN')
    plt.plot(az,10*np.log10(np.abs(pbf_vals[freq_to_plot])),label='Python')
    plt.plot(az,10*np.log10(np.abs(npbf_vals[freq_to_plot])),label='Numpy')
    plt.plot(az,10*np.log10(np.abs(nbbf_vals[freq_to_plot])),label='Numba')
    plt.legend()
    
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
        
        
        
        
        
    
    
    