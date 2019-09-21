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
            
        self.precision_types = {
                'float':np.float64,
                'complex':np.cdouble,
                'ctfloat':ctypes.c_double 
                }
        
    def set_input_types(self,*args,**kwargs):
        '''
        @brief this will change the input parameters to types in self.precision_types
        @param[in] *args - arguments to set
        @param[in/OPT] kwargs - keyword arguments to override self.precision_types
        @return tuple of args of correct types
        '''
        prec_t = self.precision_types
        for k,v in kwargs.items(): #override precision types
            self.prec_t[k] = v
        for arg in args:
            pass
    
    def load_lib(self,lib_path):
        '''
        @brief load a shared library to our class. 
        @param[in] lib_path - path to the library
        '''
        self._lib = load_ctypes_lib(lib_path)
        self._set_lib_function_types()
    
    def get_steering_vectors(self,frequency,positions,az,el):
        '''
        @brief return a set of steering vectors for a list of az/el pairs
        @param[in] frequency - frequency to calculate for
        @param[in] positions - list of xyz positions to calculate vectors for
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        '''
        az = np.deg2rad(az,dtype=self.precision_types['float'])
        el = np.deg2rad(el,dtype=self.precision_types['float'])
        positions = np.array(positions,dtype=self.precision_types['float'])
        nazel = len(az) #number of azimuth elevations
        npos = positions.shape[0] #number of positions
        steering_vectors = np.zeros((nazel,npos),dtype=self.precision_types['complex'])
        #now setup the ctypes stuff
        cfreq = ctypes.pointer(self.precision_types['ctfloat'](frequency))
        cpos = positions.ctypes; caz = az.ctypes; cel = el.ctypes
        csv = steering_vectors.ctypes
        cnazel = ctypes.pointer(ctypes.c_int(nazel))
        cnp = ctypes.pointer(ctypes.c_int(npos))
        self._lib.get_steering_vectors(cfreq,cpos,caz,cel,csv,cnp,cnazel)
        return steering_vectors
        
    def get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el):
        '''
        @brief return a numpy array of beamformed values for freq and az/el pair
        @param[in] freqs - np array of frequencies to beaform at
        @param[in] positions - xyz positions to perform at
        @param[in] weights - tapering to apply to each element
        @param[in] meas_vals - measured values at each point
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        '''
        #perform input checks
        meas_vals
        #now pass the values
        az = np.deg2rad(az,dtype=self.precision_types['float'])
        el = np.deg2rad(el,dtype=self.precision_types['float'])
        freqs = np.array(freqs,dtype=self.precision_types['float'])
        positions = np.array(positions,dtype=self.precision_types['float'])
        weights = np.array(weights,dtype=self.precision_types['complex'])
        meas_vals = np.array(meas_vals,dtype=self.precision_types['complex'])
        nazel = az.size
        nfreqs = freqs.size
        npos = positions.shape[0] #number of positions
        out_vals = np.zeros((nfreqs,nazel),dtype=self.precision_types['complex'])
        #now setup the ctypes stuff
        cfreqs = freqs.ctypes; cweights = weights.ctypes
        cmeas = meas_vals.ctypes
        cpos = positions.ctypes; caz = az.ctypes; cel = el.ctypes
        cov = out_vals.ctypes
        cnf = ctypes.pointer(ctypes.c_int(nfreqs))
        cnazel = ctypes.pointer(ctypes.c_int(nazel))
        cnp = ctypes.pointer(ctypes.c_int(npos))
        self._lib.get_beamformed_values(cfreqs,cpos,cweights,cmeas,caz,cel,cov,cnf,cnp,cnazel)
        return out_vals
        
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
    
    
    
    