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
        
        #this should be set for each inheriting class to ensure we match fortran/c subroutines
        self.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float64, #floating point default to double
                'complexfloating':np.cdouble, #defuault to complex128
                'integer':np.int32, #default to int32
                }
    
    def load_lib(self,lib_path):
        '''
        @brief load a shared library to our class. 
        @param[in] lib_path - path to the library
        '''
        self._lib = load_ctypes_lib(lib_path)
        self._set_lib_function_types()
    
    def get_steering_vectors(self,freq,positions,az,el):
        '''
        @brief return a set of steering vectors for a list of az/el pairs
        @param[in] frequency - frequency to calculate for
        @param[in] positions - list of xyz positions to calculate vectors for
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        @note template : self._lib.get_steering_vectors(freq,positions,az,el,steering_vecs_out,num_pos,num_azel)
        '''
        #change to degrees
        az = np.deg2rad(az,dtype=self.precision_types['floating'])
        el = np.deg2rad(el,dtype=self.precision_types['floating'])
        positions = np.array(positions,dtype=self.precision_types['floating'])
        num_azel = az.shape[0] #number of azimuth elevations
        num_pos = positions.shape[0] #number of positions
        steering_vecs = np.zeros((az.shape[0],positions.shape[0]),dtype=self.precision_types['complexfloating'])
        #this works...
        freq,positions,az,el,steering_vecs,num_pos,num_azel = self._set_arg_types(
                        freq,positions,az,el,steering_vecs,positions.shape[0],az.shape[0])
        arg_list = [freq,positions,az,el,steering_vecs,num_pos,num_azel]
        type_list = ['floating','floating','floating','floating','complexfloating','integer','integer']
        self._check_dict_types(arg_list,type_list)
        iargs = tuple([freq,positions,az,el,steering_vecs,num_pos,num_azel])
        #this doesnt...
        #iargs = self._set_arg_types(freq,positions,az,el,steering_vecs,positions.shape[0],az.shape[0])
        cargs = self._get_arg_ctypes(*iargs)
        self._lib.get_steering_vectors(*cargs)
        return steering_vecs
        
    def get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el):
        '''
        @brief return a numpy array of beamformed values for freq and az/el pair
        @param[in] freqs - np array of frequencies to beaform at
        @param[in] positions - xyz positions to perform at
        @param[in] weights - tapering to apply to each element
        @param[in] meas_vals - measured values at each point
        @param[in] az - np.array of azimuthal angles to calculate
        @param[in] el - np.array of elevations
        @note template : self._lib.get_beamformed_values(freqs,positions,weights,meas_vals,
                                                         az,el,out_vals,num_freqs,num_pos,num_azel)
        '''
        #set some initial values
        #degres to radians
        az = np.deg2rad(az)
        el = np.deg2rad(el)
        #sizes of our arrays
        num_azel = az.shape[0]
        num_freqs = len(freqs)
        num_pos = positions.shape[0]
        #complex output values
        out_vals = np.zeros((num_freqs,num_azel),dtype=self.precision_types['complexfloating'])
        #now set these to the correct dtypes and type check them
        #this works...
        freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel = self._set_arg_types(
                                    freqs,positions,weights,meas_vals,az,el,out_vals,
                                    num_freqs,num_pos,num_azel)
        arg_list = [freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel]
        type_list = ['floating','floating','complexfloating','complexfloating','floating',
                     'floating','complexfloating','integer','integer','integer']
        self._check_dict_types(arg_list,type_list)
        iargs = tuple([freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel])
        #this doesnt...
        #iargs = self._set_arg_types(freq,positions,az,el,steering_vecs,num_pos,num_azel)
        #get arguments as ctypes
        cargs = self._get_arg_ctypes(*iargs)
        self._lib.get_beamformed_values(*cargs)
        return out_vals
    
    def _set_arg_types(self,*args,**kwargs):
        '''
        @brief this will change the input parameters to types in self.precision_types
        @param[in] *args - arguments to set
        @param[in/OPT] kwargs - keyword arguments to override self.precision_types
        @return tuple of args of correct types
        '''
        out_args = []
        prec_t = self.precision_types
        for k,v in kwargs.items(): #override precision types
            self.prec_t[k] = v
        for arg in args:
            arg = np.array(arg) #change everything to numpy for easy editing and ctypes
            type_found = 0 #flag whether our type was found or not
            for type_name,type_val in prec_t.items(): #now find what kind of data we are and cast
                supertype = getattr(np,type_name)
                if np.issubdtype(arg.dtype,supertype):
                    out_args.append(arg.astype(type_val))
                    type_found=1
                    break
            if not type_found: #if the type was not found, raise an exception
                raise TypeError('{} not found as subtype of self.precision_types'.arg.dtype)
        return tuple(out_args)
                
    def _get_arg_ctypes(self,*args):
        '''
        @brief return args.ctype (assume all args are np.ndarray type)
        @param[in] args - arguments to get ctypes of
        @return tuple of ctype pointers
        '''
        ct = []
        for i,arg in enumerate(args):
            try:
                ct.append(arg.ctypes)
            except AttributeError:
                raise TypeError('Argument {} is of type {} not {}'.format(i,type(arg),np.ndarray))
        return tuple(ct)
    
    def _check_dict_types(self,arg_list,type_list):
        '''
        @brief check types getting the actual type from self.precision_types
        @param[in] arg_list - list of arguments to check (must be np.ndarray)
        @param[in] type_list - list of strings naming types in self.precision_types
        '''
        full_type_list = [self.precision_types[k] for k in type_list]
        check_types(arg_list,full_type_list)
        
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
        
#######################################################
# parent class for python based beamforming engines
#######################################################
class PythonBeamform(SpeedBeamform):
    '''
    @brief class to build python based beamforming engines from
    '''
    SPEED_OF_LIGHT = np.double(299792458.0)
    def __init__(self):
        '''
        @brief constructor
        '''
        super().__init__(None) #no input files
        self.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float64, #floating point default to double
                'complexfloating':np.cdouble, #defuault to complex128
                'integer':np.int32, #default to int32
                }
        
    def _get_steering_vectors(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        '''
        @brief override to utilize for python engine
        '''
        raise NotImplementedError
                
    def _get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel):
        '''
        @brief override to utilize for a python engine
        '''
        raise NotImplementedError
        
    def _get_k(self,freq,eps_r=1,mu_r=1):
        '''
        @brief get our wavenumber
        @note this part stays the same for all python implementations
        '''
        lam = self.SPEED_OF_LIGHT/np.sqrt(eps_r*mu_r)/freq
        k = 2*np.pi/lam
        return k    
    
    def _get_k_vector_azel(self,freq,az,el):
        '''
        @brief get our k vectors (e.g. kv_x = k*sin(az)*cos(el))
        @note azel here are in radians
        @note this alsow is the same for all pytho implementations
        '''
        k = self._get_k(freq,1.,1.)
        print(az,el)
        kvec = k*np.array([
                np.sin(az)*np.cos(el),
                np.sin(el),
                np.cos(az)*np.cos(el)]).transpose()
        print(np.array([
                np.sin(az)*np.cos(el),
                np.sin(el),
                np.cos(az)*np.cos(el)]).transpose())
        print(kvec)
        return kvec
    
    def _get_arg_ctypes(self,*args):
        '''
        @brief override to just return input values
        '''
        return args
    
    @property
    def _lib(self):
        '''
        @brief use this to make python engines transparent from others (like c/fortran)
        '''
        class lib_class:
            get_steering_vectors = self._get_steering_vectors
            get_beamformed_values = self._get_beamformed_values
        return lib_class
    

############################################
### Some ctypes useful functions
############################################
def check_types(arg_list,supertype_list):
    '''
    @brief check that all arguments are of correct supertype (e.g. floating v integer v complexfloating)
    @param[in] arg_list - list of arguments to check
    @param[in] supertype_list - list of np.supertypes to match (e.g. np.floating/integer/complexfloating)
    @note this is required because ctypes doesnt support complex data types
    '''
    for i,arg in enumerate(arg_list):
        try:
            stf = np.issubdtype(arg.dtype,supertype_list[i])
        except AttributeError:
            raise TypeError('Argument {} is not a {}'.format(i,np.ndarray))
        if not stf:
            raise TypeError('Argument {} has dtype {} not {}'.format(i,arg.dtype,supertype_list[i]))
            
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


########################################
### testing
########################################
    




    
    
    
    