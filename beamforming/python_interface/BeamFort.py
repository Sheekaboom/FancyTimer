'''
@author ajw
@date 8-24-2019
@brief functions and classes for creating
    python functions from fortran subroutines
'''
from ....fast_beamform import *

class BeamFort:
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    def __init__(self):
        '''
        @brief constructor
        '''
        pass
    

if __name__=='__main__':
    import os
    import numpy as np
    import ctypes
    wdir = r'C:\Users\aweis\git\pycom\beamforming\fortran\build'
    mod = ctypes.CDLL(os.path.join(wdir,'libbeamform_serial.dll'))
    mod.__beamforming_serial_MOD_fortran_mult_funct.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
    mod.__beamforming_serial_MOD_fortran_mult_funct.restype = ctypes.c_float
    ca = ctypes.c_float(3.)
    cb = ctypes.c_float(5.)
    res = mod.__beamforming_serial_MOD_fortran_mult_funct(ctypes.pointer(ca),ctypes.pointer(cb))
    print("{} * {} = {}".format(ca,cb,res))

    #numpy array multiplication in fortran
    mod.__beamforming_serial_MOD_fortran_array_mult_sub.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int)]
    aa = np.random.rand(5).astype('float32')
    ab = np.random.rand(5).astype('float32')
    ao = np.zeros_like(aa)
    caa = aa.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cab = ab.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cao = ao.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_float_p = ctypes.POINTER(ctypes.c_float)
    mod.__beamforming_serial_MOD_fortran_array_mult_sub(caa,cab,cao,ctypes.pointer(ctypes.c_int(5)))
    print(cao)
    
    #multiple math operations
    mod.__beamforming_serial_MOD_fortran_array_math_sub.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int)]
    
    n = 100000
    arr1 = np.random.rand(n).astype('float32')
    arr2 = np.random.rand(n).astype('float32')
    
    def test_mult_python(arr1,arr2):
        '''
        @brief timing with numpy iterating
        '''
        arro = np.zeros_like(arr1)
        for i in range(len(arr1)):
            arro[i] = arr1[i]*arr2[i]
        return arro
            
    def test_mult_numpy(arr1,arr2):
        arro = arr1*arr2
        return arro
    
    def test_mult_fortran(arr1,arr2):
        arro = np.zeros_like(arr1)
        mod.__beamforming_serial_MOD_fortran_array_mult_sub(arr1.ctypes.data_as(c_float_p),arr2.ctypes.data_as(c_float_p),
            arro.ctypes.data_as(c_float_p),ctypes.pointer(ctypes.c_int(arro.size)))
        return arro
    
    carr1 = arr1.ctypes.data_as(c_float_p)
    carr2 = arr2.ctypes.data_as(c_float_p)
    arro = np.zeros_like(arr1); carro = arro.ctypes.data_as(c_float_p)
    cn = ctypes.pointer(ctypes.c_int(n))
    def test_mult_fortran_no_cast(carr1,carr2,carro,cn):
        mod.__beamforming_serial_MOD_fortran_array_mult_sub(carr1,carr2,carro,cn)
        #return carro
    
    def test_math_fortran_no_cast(carr1,carr2,carro,cn):
        #This is faster in fortran! this shows that numpy is realy freakin fast, but only when the interpreter is not called
        mod.__beamforming_serial_MOD_fortran_array_math_sub(carr1,carr2,carro,cn)
        return carro
        
    def test_math_numpy(arr1,arr2):
        arro = arr1*arr2
        arro+=arr1
        arro = arr2-arro
        arro*=(arr1/arr2)
        return arro
    
    from numba import jit
    
    @jit
    def test_math_numba_jit(arr1,arr2):
      #this is actulaly a little slower than numpy even (numpy calls some base level shenanigans)
      arro = arr1*arr2
      arro+=arr1
      arro = arr2-arro
      arro*=(arr1/arr2)
      return arro  
        
        
        
        
        
        
    
    
    