'''
@author ajw
@date 8-24-2019
@brief superclass to inherit from for fast beamforming classes  
@note this is deprecated and superseded by AoaAlgorithm. There is lots of extra crap here that
   was added and did not need to be for using ctypes. Future implementations will use custom cython wrappers
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
        check_ndims(arg_list,[0,2,1,1,2,0,0])
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
        check_ndims(arg_list,[1,2,1,2,1,1,2,0,0,0])
        #this works...
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
                raise TypeError('{} not found as subtype of self.precision_types'.format(arg.dtype))
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
        @examples
        >>> myPythonBeamform._get_k(40e9)
        838.3380087806727
        '''
        lam = self.SPEED_OF_LIGHT/np.sqrt(eps_r*mu_r)/freq
        k = 2*np.pi/lam
        return k    
    
    def _get_k_vector_azel(self,freq,az,el):
        '''
        @brief get our k vectors (e.g. kv_x = k*sin(az)*cos(el))
        @note azel here are in radians
        @note this alsow is the same for all pytho implementations
        @example
        >>> import numpy as np
        >>> myPythonBeamform._get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        array([[592.7944909352411287,   0.0000000000000000, 592.7944909352411287],
               [642.2041730818633596,   0.0000000000000000, 538.8732847735016094]])
        '''
        k = self._get_k(freq,1.,1.)
        #print(az,el)
        kvec = k*np.array([
                np.sin(az)*np.cos(el),
                np.sin(el),
                np.cos(az)*np.cos(el)]).transpose()
        #print(np.array([
        #        np.sin(az)*np.cos(el),
        #        np.sin(el),
        #        np.cos(az)*np.cos(el)]).transpose())
        #print(kvec)
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
            
def check_ndims(arg_list,ndim_list):
    '''
    @brief check that all of the arguments are of the correct number of dimensions
        Have had issues in the past with numpy broadcasting and this causing issues
    @param[in] arg_list - list of arguments that are being passed
    @param[in] ndim_list - iterable of integers for the ndims each arg should have
    '''
    for i,arg in enumerate(arg_list):
        if ndim_list[i]!=np.ndim(arg):
            raise TypeError('Argument {} has {} dimensions and should have {}'.format(i,np.ndim(arg),ndim_list[i]))
            
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
import os

class SpeedBeamformUnittest:
    '''
    @brief unittest for each of our beamform types
        This will utilize precalculated values to test for steering vectors
        and for beamforming output values
    '''
    test_data_dir = './test_data'
    test_sv_vals_name = 'unittest_steering_vectors.txt'
    test_bf_vals_name = 'unittest_beamformed_values.txt'
    test_meas_vals_name = 'unittest_meas_vals.txt'

    
    def __init__(self,*args,**kwargs):
        '''
        @brief instantiate a beamforming class to test
        '''
        super().__init__(*args,**kwargs)
        self.allowed_error = 1e-10
        self.set_beamforming_class()
        
    def set_beamforming_class(self):
        '''
        @brief override this to set the correct beamforming class.
            This should also instantiate the class
        '''
        self.beamforming_class = SpeedBeamform()
        raise NotImplementedError("Please implement to set self.beamforming_class")
        
    def get_positions(self):
        '''
        @brief return testing positions
        '''
        numel = [35,35,1] #number of elements in x,y
        spacing = 2.99e8/np.max(self.get_frequency())/2 #get our lambda/2
        Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
        pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
        return pos
    
    def get_frequency(self):
        '''
        @brief return the testing frequency
        '''
        freq = 40e9
        return freq
    
    def get_angles(self):
        '''
        @brief get angles to test for 
        @return az,el
        '''
        az = np.arange(-90,90,1)
        el = np.arange(-90,90,1)
        return az,el
        
    def get_weights(self):
        '''
        @brief return our weights
        '''
        pos = self.get_positions()
        return np.ones((pos.shape[0]),dtype=np.cdouble)
    
    def check_steering_vectors(self,sv_vals):
        '''
        @brief return boolean to see whether our steering vectors are correct
        @note these values were calculated using numpy on 10-2-2019 for a 
            35x35 array at 40GHz for the angles 
            az = np.arange(-90,91,1),el=np.arange(-90,91,1).
        '''
        correct_values = self.load_test_data(SpeedBeamformUnittest.test_sv_vals_name)
        close,error = self._check_is_close(sv_vals,correct_values)
        return np.all(close),np.max(error)
        
    def test_steering_vectors(self):
        '''
        @brief test a subset of steering vectors with a known output
        '''
        az,el = self.get_angles()
        sv = self.beamforming_class.get_steering_vectors(self.get_frequency(),self.get_positions(),az,el)
        rv,max_err = self.check_steering_vectors(sv) 
        self.assertTrue(rv,msg='Max Error = {}'.format(max_err))
        
    def check_beamformed_values(self,bf_vals):
        '''
        @brief return boolean to see whether our steering vectors are correct
        @note these values were calculated using numpy on 10-2-2019 for a 
            35x35 array at 40GHz for the angles 
        '''
        correct_values = self.load_test_data(SpeedBeamformUnittest.test_bf_vals_name)
        self.plot_beamformed_vals(bf_vals)
        close,error = self._check_is_close(bf_vals,correct_values)
        return np.all(close),np.max(error)
    
    def _check_is_close(self,test_vals,ref_vals):
        '''
        @brief check whether values are within self.allowed_error of eachother (relative error)
        @note this also will return the relative error
        @param[in] test_vals - what vals to test closeness of
        @param[in] ref_vals - ref vals to test closeness to (used for relative calculation)
        @return bool array of isclose, and the array of the error between the two
        '''
        tf = np.isclose(test_vals,ref_vals,rtol=self.allowed_error,atol=0)
        err = np.abs(test_vals-ref_vals)/np.abs(ref_vals)
        return tf,err
    
    def plot_beamformed_vals(self,bf_vals):
        import matplotlib.pyplot as plt
        az,el = self.get_angles()
        bf_data = 10*np.log10(np.abs(bf_vals[0]))
        plt.plot(az,bf_data,label='Unittest Beamformed')
        baseline = 10*np.log10(np.abs(self.load_test_data(SpeedBeamformUnittest.test_bf_vals_name)))
        plt.plot(az,baseline,label='Unittest Baseline')
    
    def test_beamformed_values(self):
        '''
        @brief test a subset of steering vectors with a known output
        '''
        bf_vals = self.get_beamformed_values()
        rv,max_err = self.check_beamformed_values(bf_vals)
        self.assertTrue(rv,msg='Max Error = {}'.format(max_err))
        
    def get_beamformed_values(self):
        '''
        @brief makes for easier getting of the outputs
        '''
        az,el = self.get_angles()
        freqs = [self.get_frequency()]
        pos = self.get_positions()
        weights = self.get_weights()
        meas_vals = self.load_test_data(SpeedBeamformUnittest.test_meas_vals_name)
        meas_vals = meas_vals.reshape((1,-1)) # this value is squeezed so we have to unsqeeze else its treated as a scalar in beamforming
        return self.beamforming_class.get_beamformed_values(freqs,pos,weights,meas_vals,az,el)
    
    def load_test_data(self,fname):
        '''
        @brief method to load all of our test data to compare against
        @note this uses self.test_data_dir for directory of fname. this also 
            assumes complex 128 data read and write
        '''
        return np.loadtxt(os.path.join(self.test_data_dir,fname),
                          dtype=np.cdouble,comments='#')
    
    ########## constants for correct values #################
    # it may be best to just load thest from a file in the future...
    '''
    @note these values were calculated using numpy on 10-1-2019 for a 
            35x35 array at 40GHz for the angles 
            az = np.arange(-90,91,10),el=np.zeros_like(az).
            beamformed_meas_vals created with plane wave at [0,0] and [45,20] ([az,el])
    @todo put into a file and load from there
    '''
    
if __name__=='__main__':
    import doctest
    doctest.testmod(globs={'myPythonBeamform':PythonBeamform()})
    
    
    