'''
@author ajw
@date 8-24-2019
@brief functions and classes for creating
    python functions from fortran subroutines
'''
import numpy as np
import os

from SpeedBeamform import SpeedBeamform
from SpeedBeamform import get_ctypes_pointers
import ctypes
wdir = os.path.dirname(os.path.realpath(__file__))


serial_fortran_library_path = os.path.join(wdir,'../fortran/build/libbeamform_serial.dll')
class SerialBeamformFortran(SpeedBeamform):
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    def __init__(self):
        '''
        @brief constructor
        '''
        super().__init__(serial_fortran_library_path)
        
    def get_steering_vectors(self,frequency,positions,az,el):
        '''
        @brief get steering vectors with inputs same as super().get_steering_vectors()
        '''
        az = np.array(az)
        el = np.array(el)
        positions = np.array(positions)
        nazel = len(az) #number of azimuth elevations
        np = positions.shape[0] #number of positions
        self._lib.get_steering_vectors
        
        
    def _set_lib_function_types(self):
        '''
        @brief set the function types for our library functions when we load
        @note have not found a way to nicely define a complex argtype
        '''
        # get_steering_vectors() function
        self._lib.get_steering_vectors.argtypes = [ctypes.POINTER(ctypes.c_float) for i in range(4)]

if __name__=='__main__':
    n = 5
    cnp = ctypes.pointer(ctypes.c_int(n))
    mysbf = SerialBeamformFortran()
    
    #test array multiply
    arr1 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arr2 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arro = np.zeros_like(arr1)
    mysbf.test_complex_array_multiply(arr1.ctypes,arr2.ctypes,arro.ctypes,cnp)
    nparro = arr1*arr2
    
    #test matrix multiply
    mat1 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    mat2 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    mato = np.zeros_like(mat1)
    
        
        
        
        
        
        
    
    
    