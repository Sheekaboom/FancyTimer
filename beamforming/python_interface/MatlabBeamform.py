'''
@author ajw
@date 10-2-2019
@brief Python bindings for MATLAB beamforming
'''
import numpy as np
import os

from samurai.base.SamuraiMatlab import SamuraiMatlab
from pycom.beamforming.python_interface.SpeedBeamform import PythonBeamform

fdir = os.path.realpath(os.path.dirname(__file__))

#######################################################
# parent class for MATLAB based beamforming engines
#######################################################
class MatlabBeamform(PythonBeamform):
    '''
    @brief class to build matlab based beamforming engines
    @note PythonBeamform is inherited from here because matlab cant pass by reference
        Therefore we cannot match the typical SpeedBeamform Function Prototype
    '''
    matlab_root_dir = os.path.join(fdir,'../MATLAB')
    matlab_generic_dir = os.path.join(matlab_root_dir,'generic') #generic functions

    def __init__(self,code_dir,**kwargs):
        '''
        @brief constructor
        @param[in] code_dir - directory that contains matlab functions
        @param[in/OPT] - kwargs keyword arguments as follows:
            engine - optional matlab engine if already started
        '''
        super().__init__() #no input files
        self._matlab_lib = SamuraiMatlab(**kwargs)
        self._init_code_dirs([code_dir])
        self.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float64, #floating point default to double
                'complexfloating':np.cdouble, #defuault to complex128
                'integer':np.int32, #default to int32
                }
        
    def _init_code_dirs(self,rel_dir_list):
        '''
        @brief add our code paths to the matlab engine path
        @param[in] rel_dir_list - path to code relative to MatlabBeamform.matlab_root_dir
        '''
        self._matlab_lib.addpath(os.path.realpath(MatlabBeamform.matlab_generic_dir))
        for rel_dir in rel_dir_list:
            abs_dir = os.path.join(MatlabBeamform.matlab_root_dir,rel_dir)
            self._matlab_lib.addpath(os.path.realpath(abs_dir))
        
        
#######################################################
# parent class for MATLAB based beamforming engines
#######################################################
class MatlabSerialBeamform(MatlabBeamform):
    '''
    @brief serial matlab beamforming class
    '''
    def __init__(self,**kwargs):
        '''
        @brief constructor
        @param[in/OPT] - kwargs keyword arguments as follows:
            engine - optional matlab engine if already started
        '''
        code_dir = os.path.join(MatlabBeamform.matlab_root_dir,'serial')
        super().__init__(code_dir,**kwargs)
        
    def _get_steering_vectors(self,freq,positions,az,el,steering_vecs_out,num_pos,num_azel):
        '''
        @brief override to utilize for python engine
        '''
        freq = float(freq) #cannot be numpy scalar must be double. Float is double in python i guess
        steering_vecs_out[:,:] = self._matlab_lib.get_steering_vectors(freq,positions,az,el)

    def _get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el,out_vals,num_freqs,num_pos,num_azel):
        '''
        @brief override to utilize for a python engine
        '''
        out_vals[:,:] = self._matlab_lib.get_beamformed_values(freqs,positions,weights,meas_vals,az,el)
        
    
#######################################################
# Testing for all the files
#######################################################    
if __name__=='__main__':
    
    from pycom.beamforming.python_interface.SpeedBeamform import SpeedBeamformUnittest
    import unittest
    
    class MatlabSerialUnittest(SpeedBeamformUnittest,unittest.TestCase):
        def set_beamforming_class(self):
            self.beamforming_class = MatlabSerialBeamform()
    
       
    test_class_list = [MatlabSerialUnittest]
    tl = unittest.TestLoader()
    #suite = unittest.TestSuite(tl.loadTestsFromTestCase(MatlabSerialUnittest))
    suite = unittest.TestSuite([tl.loadTestsFromTestCase(mycls) for mycls in test_class_list])
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    #start an isntance for testing
    #mymb = MatlabSerialBeamform()
    
    
    