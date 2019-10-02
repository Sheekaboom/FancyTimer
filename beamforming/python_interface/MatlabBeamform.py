'''
@author ajw
@date 10-2-2019
@brief Python bindings for MATLAB beamforming
'''
import numpy as np
import os

from samurai.base.SamuraiMatlab import SamuraiMatlab
from pycom.beamforming.python_interface.SpeedBeamform import SpeedBeamform

fdir = os.path.realpath(os.path.dirname(__file__))

#######################################################
# parent class for MATLAB based beamforming engines
#######################################################
class MatlabBeamform(SpeedBeamform):
    '''
    @brief class to build python based beamforming engines from
    '''
    matlab_root_dir = os.path.join(fdir,'../MATLAB')
    matlab_generic_dir = os.path.join(matlab_root_dir,'generic') #generic functions

    def __init__(self,code_dir):
        '''
        @brief constructor
        @param[in] code_dir - directory that contains matlab functions
        '''
        super().__init__(None) #no input files
        self._lib = SamuraiMatlab()
        self.precision_types = { #these match numpy supertype names (e.g. np.floating)
                'floating':np.float64, #floating point default to double
                'complexfloating':np.cdouble, #defuault to complex128
                'integer':np.int32, #default to int32
                }
        
#######################################################
# parent class for MATLAB based beamforming engines
#######################################################
class MatlabSerialBeamform:
    
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
    suite = unittest.TestSuite([tl.loadTestsFromTestCase(mycls) for mycls in test_class_list])
    unittest.TextTestRunner(verbosity=2).run(suite)