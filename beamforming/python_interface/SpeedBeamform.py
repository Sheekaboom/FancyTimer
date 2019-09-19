'''
@author ajw
@date 8-24-2019
@brief superclass to inherit from for fast beamforming classes
'''

import ctypes
import numpy as np

class SpeedBeamform:
    '''
    @brief superclass template to create fast beamforming python interfaces
    '''
    def __init__(self,lib_path=None):
        '''
        @brief constructor
        @param[in/OPT] lib_path - path to shared library to load
        '''
        if lib_path is not None:
            self.load_lib(lib_path)
    
    def load_lib(self,lib_path):
        '''
        @brief load a shared library to our class. 
        @param[in] lib_path - path to the library
        '''
        self._lib = load_ctypes_lib(lib_path)
        self._set_lib_function_types()
    
    def get_steering_vectors(frequency,positions,az,el):
        '''
        @brief return a set of steering vectors for a list of az/el pairs
        @param[in] frequency - frequency to calculate for
        @param[in] positions - list of xyz positions to calculate vectors for
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        '''
        raise NotImplementedError
        
    def get_beamformed_values(freqs,positions,weights,meas_vals,az,el):
        '''
        @brief return a numpy array of beamformed values for freq and az/el pair
        @param[in] freqs - np array of frequencies to beaform at
        @param[in] positions - xyz positions to perform at
        @param[in] weights - tapering to apply to each element
        @param[in] meas_vals - measured values at each point
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        '''
        raise NotImplementedError
        
    def _set_lib_function_types(self):
        '''
        @brief override this in subclasses to set types of fucntions.
            This gets called with load_lib. for Ctypes set argtypes and restype
            property of each function
        @note if not overriden no types will be defined
        '''
        pass
        
    def __getattr__(self,attr):
        '''
        @brief try and call attribute from _lib if not here
        '''
        try:
            return getattr(self._lib,attr)
        except AttributeError:
            raise AttributeError(attr)
        
        


############################################
### Some ctypes useful functions
############################################
def load_ctypes_lib(lib_path,**kwargs):
    '''
    @brief load a ctypes library
    @param[in] lib_path - path to library to load
    @param[in/OPT] kwargs - other arguments passed to ctypes.CDLL()
    '''
    lib = ctypes.CDLL(lib_path,**kwargs)
    return lib

def get_ctypes_pointers(*args,**kwargs):
    '''
    @brief get all arguments as ctypes pointers. return tuple of arg pionters
        this is useful for passing data by reference
    @param[in] args - arguments to get pointers of
    @param[in] kwargs - keywords passed to ctypes.pointer()
    '''
    args_p = tuple(ctypes.pointer(a,**kwargs) for a in args)
    return args_p
    
    
    
    