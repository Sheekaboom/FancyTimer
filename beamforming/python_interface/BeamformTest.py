'''
@author ajw
@date 10-1-2019
@brief testing for different beamforming algorithms
'''
import numpy as np
from collections import OrderedDict
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumpy
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumba
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformFortran
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformPython

class BeamformTest(OrderedDict): #extending ordereddict nicely allows us to print out a full test
    '''
    @brief class to test all of the beamforming code
        This will compare between all of our codes. Not for a single given value.
        This will also compare speeds of the values
    '''
    def __init__(self,*args,**kwargs):
        '''
        @brief initialize TestCase then add dictionary of classes
        '''
        self._init_beamform_classes()
        self._init_angles()
        self._init_frequencies()
        self._init_positions()
        self['baseline_key'] = 'NUMPY' #baseline to compare to
        self['allowed_error'] = 1e-12 #maximum error allowed between algorithms
        
    def _init_beamform_class_dict(self):
        '''
        @brief initialize all of our beamforming classes
        '''
        self['beamform_class_dict'] = {
                'NUMPY':SerialBeamformNumpy,
                'NUMBA':SerialBeamformNumba,
                'FORTRAN':SerialBeamformFortran
                }
        #self._beamform_class_dict['PYTHON'] = SerialBeamformPython
        
    def _init_beamform_classes(self):
        '''
        @brief initialize self._beamformers
        '''
        self._init_beamform_class_dict()
        self._beamformers = {k:v() for k,v in self['beamform_class_dict'].items()}
        
    def _init_positions(self):
        '''
        @brief initialize element lcoations test a 2D planar array
        '''
        numel = [35,35,1] #number of elements in x,y
        spacing = 2.99e8/np.max(self['frequencies'])/2 #get our lambda/2
        Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
        pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
        self['positions'] = pos
        
    def _init_angles(self):
        '''
        @brief initialize angles to beamform across
        '''
        azl = np.arange(-90,90,1)
        ell = np.arange(-90,90,1)
        #ell = 0
        AZ,EL = np.meshgrid(azl,ell)
        self['az'] = AZ.flatten()
        self['el'] = EL.flatten()
        
    def _init_frequencies(self):
        '''
        @brief frequencies to test across
        '''
        self['frequencies'] = np.arange(26.5e9,27e9,.1e9)
        
    def test_steering_vectors(self):
        '''
        @brief calls get_steering_vectors and compares outputs to self.baseline_key
            and ensure we are within self.allowed_error
        '''
        np.set_printoptions(precision=16,floatmode='fixed')
        print("STEERING VECTOR EQUALITY CHECK:")
        self._test_equality('get_steering_vectors',self['frequencies'][0],self['positions'],self['az'],self['el'])

    def _test_equality(self,funct_name,*args):
        '''
        @brief test the equality of return values from a function contained in
            self._beamformers[name]
        @param[in] funct_name - function (method) name to test equality of rv from
            must be contained in all self._beamformers[name] values
        @param[in] *args - arguments to pass to funct
        @todo add timing into here
        '''
        base_val = getattr(self._beamformers[self['baseline_key']],funct_name)(*args)
        for k,v in self._beamformers.items():
            cur_val = getattr(v,funct_name)(*args)
            val_equal = np.isclose(cur_val,base_val,rtol=self['allowed_error'],atol=0)
            all_equal = np.all(val_equal)
            print("{}: {}".format(k,all_equal))
            if not all_equal:
                neq_tf = np.invert(val_equal) #true false array of where things arent equal
                err = np.abs(cur_val-base_val)/np.abs(base_val)
                print("    Non-equal Values:  {}%".format(np.sum(neq_tf)/len(neq_tf)*100))
                print("    Max  Error      :  {} ".format(np.max(err)))
                print("    Mean Error      :  {} ".format(np.mean(err)))
                print("    Max Error Index :  {} ".format(np.argmax(err)))
            
      
   # def test_beamform
    
    
    
    
    
    
    
if __name__ == '__main__':
    bftest = BeamformTest()
    bftest.test_steering_vectors()