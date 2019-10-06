'''
@author ajw
@date 10-1-2019
@brief testing for different beamforming algorithms
'''
import numpy as np
import timeit

try:
    from samurai.base.SamuraiDict import   SamuraiDict
except ModuleNotFoundError:
    from collections import OrderedDict as SamuraiDict
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumpy
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformNumba
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformFortran
from pycom.beamforming.python_interface.SerialBeamform import SerialBeamformPython
#from pycom.beamforming.python_interface.MatlabBeamform import SerialBeamformMatlab

class BeamformTest(SamuraiDict): #extending ordereddict nicely allows us to print out a full test
    '''
    @brief class to test all of the beamforming code
        This will compare between all of our codes. Not for a single given value.
        This will also compare speeds of the values
    '''
    def __init__(self,*args,**kwargs):
        '''
        @brief initialize TestCase then add dictionary of classes
        '''
        super().__init__(*args,**kwargs)
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
        self['beamform_class_dict']['MATLAB'] = SerialBeamformMatlab
       #self['beamform_class_dict']['PYTHON'] = SerialBeamformPython
        
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
        
    def _init_weights(self):
        '''
        @brief initialize the weights for testing
        '''
        self['weights'] = np.ones(self['positions'].shape[0],dtype=np.cdouble)
        
    def _init_meas_vals(self):
        '''
        @brief initialize measured values
        '''
        print("Calculating Synthetic Measured Values")
        #get the steering vectors to use
        val_list = []
        inc_az = [0,45]; inc_el = [0,20]
        for f in self['frequencies']:
            mv = self._beamformers[self['baseline_key']].get_steering_vectors(f,
                                          self['positions'],inc_az,inc_el)
            val_list.append(mv)
        self['meas_vals'] = np.array(val_list).sum(axis=1)
        
    def _init_frequencies(self):
        '''
        @brief frequencies to test across
        '''
        self['frequencies'] = np.arange(26.5e9,27e9,.1e9)
        
    def test_steering_vectors(self,**kwargs):
        '''
        @brief calls get_steering_vectors and compares outputs to self.baseline_key
            and ensure we are within self.allowed_error
        '''
        np.set_printoptions(precision=16,floatmode='fixed')
        print("STEERING VECTOR EQUALITY CHECK:")
        sd = self._test_equality_and_time('get_steering_vectors',self['frequencies'][0],
                                          self['positions'],self['az'],self['el'],**kwargs)
        self['steering_vector_test'] = sd
        
    def test_beamformed_values(self,**kwargs):
        '''
        @brief calls and tests get_beamformed_values outputs to self.baseline_key
            and ensure we are within self.allowed_error
        '''
        self._init_weights()
        self._init_meas_vals()
        np.set_printoptions(precision=16,floatmode='fixed')
        print("BEAMFORMED VALUE EQUALITY CHECK:")
        sd = self._test_equality_and_time('get_beamformed_values',self['frequencies'],
                                          self['positions'],self['weights'],
                                          self['meas_vals'],self['az'],self['el'],**kwargs)
        self['beamformed_values_test'] = sd

    def _test_equality_and_time(self,funct_name,*args,**kwargs):
        '''
        @brief test the equality of return values from a function contained in
            self._beamformers[name]
        @param[in] funct_name - function (method) name to test equality of rv from
            must be contained in all self._beamformers[name] values
        @param[in] *args - arguments to pass to funct
        @param[in/OPT] **kwargs - keyword arguments. num_reps will be used for timing
        @todo add timing into here
        '''
        bf_stat_list = [] #list of statistics dictionaries
        print("Getting Baseline {}: ".format(self['baseline_key']),end='')
        base_time,base_val = fancy_timeit(lambda: getattr(self._beamformers[self['baseline_key']],funct_name)(*args),**kwargs)
        print("Done")
        for k,v in self._beamformers.items():
            stat_dict = SamuraiDict()
            print("{}: ".format(k),end='')
            if k is not self['baseline_key']: #dont rerun baseline
                cur_time,cur_val = fancy_timeit(lambda: getattr(v,funct_name)(*args)) #run the function
            else:
                cur_val = base_val; cur_time = base_time
            #check if were within our margin of error
            val_equal = np.isclose(cur_val,base_val,rtol=self['allowed_error'],atol=0)
            #now save the statistics
            err = np.abs(cur_val-base_val)/np.abs(base_val)
            stat_dict['Engine']           = k
            stat_dict['allowed_error']  = self['allowed_error']
            stat_dict['percentage_within_error'] = np.sum(val_equal[:,0])/len(val_equal)*100
            stat_dict['max_error']        = np.max(err)
            stat_dict['mean_error']       = np.mean(err)
            stat_dict['max_error_idx']    = np.argmax(err)
            #set timing info
            cur_time.set_speedup(base_time)
            stat_dict['timing'] = cur_time
            all_equal = np.all(val_equal)
            print("{}".format(all_equal))
            #if not all_equal:
            print(stat_dict.tostring())
            bf_stat_list.append(stat_dict)   
        return bf_stat_list
        
  
class FancyTimerStats(SamuraiDict):
    '''
    @brief class to hold and manipulate data from fancy_timeit function
    '''
    def __init__(self,time_list,*args,**kwargs):
        '''
        @brief constructor
        @param[in] time_list - list of times that were measured for our repeats
        '''
        super().__init__(*args,**kwargs)
        self._calc_stats(time_list)
        
    def _calc_stats(self,time_list):
        '''@brief calculate and store time stats'''
        time_list = np.array(time_list)
        self['mean'] = np.mean(time_list)
        self['stdev']= np.std(time_list)
        self['count']= len(time_list)
        self['min']  = np.min(time_list)
        self['max']  = np.max(time_list)
        self['range']= np.ptp(time_list)
        
    def set_speedup(self,base_fts):
        '''
        @brief add speedup statistics to the timer
        @param[in] base_fts - baseline FancyTimerStats class 
        '''
        self['speedup'] = base_fts['mean']/self['mean']


fancy_template = '''
def inner(_it, _timer{init}):
    {setup}
    time_list = []
    for _i in _it:
        _t0 = _timer()
        retval = {stmt}
        _t1 = _timer()
        time_list.append(_t1-_t0) #append the time to run
    return time_list, retval
'''    
def fancy_timeit(mycallable,num_reps=3):
    '''
    @brief easy timeit function that will return the results of a function
        along with a dictionary of timing statistics
    @param[in] mycallable - callable statement to time
    @param[in/OPT] num_reps - number of repeats for timing and statistics
    '''
    
    timeit.template = fancy_template #set the template
    ft = timeit.Timer(mycallable)
    tl,rv = ft.timeit(number=num_reps)
    return FancyTimerStats(tl),rv
    
    
    
    
if __name__ == '__main__':
    bftest = BeamformTest()
    bftest['frequencies'] = [40e9]
    bftest.test_steering_vectors()
    #bftest.test_beamformed_values()
    
    '''
    #timer testing
    def test(x,y):
        return x+y
    def test2(x,y):
        return x**y
    fts,rv = fancy_timeit(lambda: test(4,5),100)
    fts2,rv2 = fancy_timeit(lambda: test2(4,5),100)
    fts.set_speedup(fts2)
    print(fts.tostring())
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    