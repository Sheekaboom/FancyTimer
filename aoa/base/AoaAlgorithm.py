'''
@brief some base classes and functions for AoA estimation algorithms.
This will have relatively fast implementation of some things (like steering vectors)  
@author Alec Weiss  
@date  11/2019  
'''

import numpy as np
import cmath
import numpy as np
from numba import vectorize
import scipy.interpolate #for finding nearest incident

class AoaAlgorithm:
    '''@brief this is a base class for creating aoa algorithms'''
    SPEED_OF_LIGHT = np.double(299792458.0)
    def __init__(*args,**kwargs):
        '''@brief constructor. Dont do anything right now'''
        pass
    
    #%% Lots of methods that should be defined by subclasses
    @classmethod
    def get_k(self,freq,eps_r=1,mu_r=1):
        '''
        @brief get our wavenumber  
        @note this part stays the same for all python implementations  
        @example  
        >>> AoaAlgorithm.get_k(40e9,1,1)  
        838.3380087806727  
        '''
        lam = self.SPEED_OF_LIGHT/np.sqrt(eps_r*mu_r)/freq
        k = 2*np.pi/lam
        return k 
    
    @classmethod
    def get_k_vector_azel(self,freq,az,el,**kwargs):
        '''
        @brief get our k vectors (e.g. kv_x = k*sin(az)*cos(el))  
        @param[in] freq - frequency in hz  
        @param[in] az - azimuth in radians (0 is boresight)  
        @param[in] el - elevation in radians (0 boresight)  
        @param[in] kwargs - keyword args passed to get_k  
        @example  
        >>> import numpy as np  
        >>> AoaAlgorithm.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))  
        array([[592.7944909352411287,   0.0000000000000000, 592.7944909352411287],  
               [642.2041730818633596,   0.0000000000000000, 538.8732847735016094]])  
        '''
        k = self.get_k(freq,**kwargs)
        k = k.reshape(-1,1,1)
        kvec = np.array([
                np.sin(az)*np.cos(el),
                np.sin(el),
                np.cos(az)*np.cos(el)]).transpose()
        kvec = k*kvec[np.newaxis,...]
        return kvec
    
    @classmethod
    def get_steering_vectors(self,freqs,pos,az,el,**kwargs):
        '''
        @brief return a set of steering vectors for a list of az/el pairs  
        @param[in] freqs - frequencies to calculate for  
        @param[in] pos - list of xyz positions of points in the array  
        @param[in] az - np.array of azimuthal angles in radians  
        @param[in] el - np.array of elevations in radians  
        @param[in/OPT] kwargs - keyword values as follows:
            dtype - complex data type to use (e.g. np.cdouble)
        @return np.ndarray of size (len(freqs),len(az),len(pos))  
        '''  
        if np.ndim(freqs)<1: freqs = np.asarray([freqs])
        if np.ndim(az)<1:    az = np.asarray([az])
        if np.ndim(el)<1:    el = np.asarray([el])
        if np.ndim(pos)<1:   pos = np.asarray([pos])
        options = {}
        options['dtype'] = np.cdouble
        for k,v in kwargs.items():
            options[k] = v
        freqs = np.asarray(freqs) #change to nparray
        kvecs = self.get_k_vector_azel(freqs,az,el)
        steering_vecs_out = np.ndarray((len(freqs),len(az),len(pos)),dtype=options['dtype'])
        for fn in range(len(freqs)):
            steering_vecs_out[fn,...] = self.vector_exp_complex(np.matmul(pos,kvecs[fn,...].transpose())).transpose()
        return steering_vecs_out
    
    @classmethod
    def calculate(self,freqs,pos,meas_vals,az,el,**kwargs):
        '''
        @brief calculate and return values o the algorithm for the given input  
        @param[in] freqs - np array of frequencies to calculate at  
        @param[in] pos - xyz positions to perform at  
        @param[in] meas_vals - measured values at each position and frequency of shape (len(freqs),len(pos))  
        @param[in] az - np.array of azimuthal angles to calculate radians  
        @param[in] el - np.array of elevations radians  
        @param[in] kwargs - keyword args as follows:  
            - weights - list of complex weightings to add to each position (taperings)  
        '''
        raise NotImplementedError
    
    #%% numba vectorize operations
    from numba import complex128,complex64
    @vectorize([complex128(complex128),
                complex64 (complex64 )],target='parallel')
    def vector_exp_complex(vals):
        '''@brief numba compex exponential'''
        return cmath.exp(-1j*vals)
    
    @vectorize([complex128(complex128,complex128),
                complex64 (complex64 ,complex64 )],target='parallel')
    def vector_mult_complex(a,b):
        '''@brief numba complex vector multiplication''' 
        return a*b

    @vectorize([complex128(complex128,complex128,complex128),
                complex64 (complex64 ,complex64 ,complex64 )],target='cpu')
    def vector_beamform(weights,meas_vals,sv_no_exp):
        '''@brief numba complex beamform multiplication'''
        return weights*meas_vals*cmath.exp(-1j*sv_no_exp)

#%% Synthetic data creation
    @classmethod
    def synthesize_data(self,freq,pos,az,el,mag=1,**kwargs):
        '''
        @brief create synthetic data for an array. This will generate synthetic
            data assuming incident plane waves from az,el angle pairs  
        @param[in] freq - frequency to create data for in hz  
        @param[in] pos - position of elements in meters  
        @param[in] az - azimuth of incident plane wave(s) (radians)  
        @param[in] el - elevation of incident plane wave(s) (radians)  
        @param[in/OPT] mag - magnitude of plane wave in linear (default 1)  
        @param[in/OPT] kwargs - keyword arguments as follows:    
            - snr - signal to noise ratio of the signal (default inf (no noise)).
                    This is in dB compared to 10*log10(mag)  
            - noise_funct - noise function to create noise (default np.random.randn)  
            - dtype - data type (np.csingle or np.cdouble) to return data as
        @return array of synthetic data corresponding to each element in pos. rv[freqs][pos]  
        '''
        #parse inputs
        options = {}
        options['snr'] = np.inf
        options['noise_funct'] = np.random.randn
        options['dtype'] = np.cdouble
        for k,v in kwargs.items():
            options[k] = v
        #create the synthetic data
        kvecs = self.get_k_vector_azel(freq,az,el)
        synth_data = np.array([self.vector_exp_complex(-np.matmul(pos,kv.T)) for kv in kvecs],dtype=options['dtype']) #go through all az,el angles
        #add our magnitudes
        mag = np.reshape(mag,(1,1,-1)) #allow for a list of magnitudes here also
        synth_data *= mag
        #now sum across all incidences
        synth_data = synth_data.sum(axis=-1)
        #now add the noise
        snr_lin = 10**(options['snr']/10)
        noise_mag = np.max(np.abs(synth_data))/snr_lin #max magnitude
        noise_vec = noise_mag*(options['noise_funct'](*synth_data.shape) #create our noise
                          +1j*options['noise_funct'](*synth_data.shape))
        synth_data += noise_vec #add to our data
        return synth_data

#%% unit testing
import unittest

class TestAoaAlgorithm(unittest.TestCase):
    '''
    @brief basic unittesting of AoaAlgorithm object. To use this, inherit from
        this class and override self.set_options to set values in self.options dictionary  
    @note the following keys should be set in the self.options dict:    
        - aoa_class - angle of arrival algorithm class to test  
        - allowed_error - amount of allowable error in comaprison to other data (default 1e-10)  
        - calculated_data_path - path to data that can be loaded in with np.loadtxt to compare output to  
    @example   
        from pycom.aoa.base.AoaAlgorithm import TestAoaAlgorithm  
        import MusicAlgorithm #assuming its in this directory  
        TestMusicAlgorithm(TestAoaAlgorithm):  
            def set_options(self):  
                self.aoa_class = MusicAlgorithm  
    '''
    
    def __init__(self,*args,**kwargs):
        '''@brief initiate the class to test'''
        super().__init__(*args,**kwargs)
        self.options = {}
        self.options['calculated_data_path'] = None
        self.options['allowed_error'] = 1e-10
        self.options['aoa_class'] = None
        self.options['plotter'] = None #plot library to use
        self.verification_dict = { #dictionary to verify options. If not exist v is None
                'allowed_error': lambda v: v is not None and v>0,
                'aoa_class'    : lambda v: v is not None,
                }
        self.set_options() #run the overriden set options class
        self.verify_options()
        
    def set_options(self):
        '''@brief set options for the test class'''
        self.options['aoa_class'] = AoaAlgorithm
        
    def verify_options(self):
        '''@brief verify provided options are good'''
        for k,ver_fun in self.verification_dict.items():
            option = self.options.get(k,None)
            if not ver_fun(option): #if verification fails, raise an exception
                raise Exception("{} does not pass option verification for {}".format(option,k))
        #otherwise we are successful and dont do antyhing
    
    def test_get_k(self):
        '''@brief test getting the wavenumber'''
        myaoa = self.options['aoa_class']() #test both static and object
        o_ans = myaoa.get_k(40e9,1,1);
        s_ans = self.options['aoa_class'].get_k(40e9,1,1);
        expected_ans = 838.3380087806727
        self.assertEqual(o_ans,expected_ans)
        self.assertEqual(s_ans,expected_ans)
        
    def test_get_k_vector_azel(self):
        '''@brief test getting k vector in azel'''
        myaoa = self.options['aoa_class']()
        o_ans = myaoa.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        s_ans = AoaAlgorithm.get_k_vector_azel(40e9,np.deg2rad([45, 50]),np.deg2rad([0,0]))
        expected_ans = np.array([[592.7944909352411,   0.000, 592.7944909352411],
                                 [642.2041730818634,   0.000, 538.8732847735016]])
        self.assertTrue(np.all(np.squeeze(o_ans)==expected_ans))
        self.assertTrue(np.all(np.squeeze(s_ans)==expected_ans))
        
    def test_calculate_against_data(self):
        '''
        @brief test self.options['aoa_class'].caculate against data from 
            self.options['calculated_data_path']. This is only performed if
            self.options['calculated_data_path'] is not None. Otherwise we pass  
        '''
        if self.options['calculated_data_path'] is not None:
            data_from_file = np.loadtxt(self.options['calculated_data_path'],dtype=np.cdouble)
            calculated_data = self.options['aoa_class'].calculate(self.freqs
                                          ,self.positions,meas_vals,az,el,weights=self.weights)
        else:
            pass #othwerise lets just pass the test
            
#%% initialization for plotting 
    def _init_plotting(self):
        if self.options['plotter'] is None:
            import plotly.graph_objects as go 
            self.options['plotter'] = go 
        

#%% properties for testing aoa   
    @property
    def freqs(self):
        '''@brief frequencies to test at'''
        return np.arange(27.5e9,31e9,1e9)        
    
    @property
    def meas_vals(self):
        '''@brief create measured values from our inicident and positions and freqs'''
        kwargs = {}
        az,el = self.incident_angles
        vals = self.options['aoa_class'].synthesize_data(self.freqs
                          ,self.positions,az,el,mag=1,**kwargs)
        return vals
    
    @property
    def angles(self):
        '''@brief get angles to test at in radians'''
        az = np.linspace(-np.pi/2,np.pi/2,181)
        el = np.linspace(-np.pi/2,np.pi/2,181)
        return az,el
    
    @property
    def ANGLES(self):
        '''@brief get meshgridded angles'''
        az,el = self.angles
        AZ,EL = np.meshgrid(az,el)
        return AZ,EL
    
    @property
    def incident_angles(self):
        '''@brief set our angles of incidence of plane waves (for synthetic testing)'''
        azi = np.array([-np.pi/3,-np.pi/8,np.pi/4])
        eli = np.array([np.pi/5,0        ,np.pi/3])
        az,el = self.angles
        faz = scipy.interpolate.interp1d(az,az,'nearest') #find the nearest values
        fel = scipy.interpolate.interp1d(el,el,'nearest')
        azi = faz(azi)
        eli = fel(eli)
        return azi,eli
    
    @property
    def weights(self):
        '''@brief return our weights'''
        return np.ones((self.positions.shape[0]),dtype=np.cdouble)
    
    @property
    def positions(self):
        '''@brief get testing positions'''
        numel = [35,35,1] #number of elements in x,y
        spacing = 2.99e8/np.max(self.freqs)/2 #get our lambda/2
        Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
        pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
        return pos
    
    def plot_2d_calc(self):
        '''@brief plot 2D calculated data with dots for correct reference'''
        self._init_plotting()
        go = self.options['plotter']
        myaoa = self.options['aoa_class']()
        AZ,EL = self.ANGLES
        pos = self.positions
        freqs = self.freqs
        meas_vals = self.meas_vals
        out_vals = myaoa.calculate(freqs,pos,meas_vals,AZ.flatten(),EL.flatten())
        OV = out_vals[-1].reshape(AZ.shape) #just take last frequency
        fig = go.Figure(go.Surface(x=AZ,y=EL,z=10*np.log10(np.abs(OV))))
        azi,eli = self.incident_angles
        fig.add_trace(go.Scatter3d(x=azi,y=eli,z=np.zeros_like(azi),mode='markers'))#,marker=dict(size=np.ones_like(azi)*3)
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #surf = ax.plot_surface(AZ, EL, 10*np.log10(np.abs(OV)), cmap=cm.coolwarm,
        #               linewidth=0, antialiased=False)
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()
        fig.show(renderer='browser')
        return fig
    
#%% properties for testing aoa 1D. freqs same as 2D 
    @property
    def meas_vals_1d(self):
        '''@brief create measured values from our inicident and positions and freqs'''
        kwargs = {}
        az,el = self.incident_angles_1d
        vals = self.options['aoa_class'].synthesize_data(self.freqs
                          ,self.positions_1d,az,el,mag=1,**kwargs)
        return vals
    
    @property
    def angles_1d(self):
        '''@brief get angles to test at in radians'''
        az = np.linspace(-np.pi/2,np.pi/2,181)
        el = np.zeros_like(az)
        return az,el
    
    @property
    def incident_angles_1d(self):
        '''@brief set our angles of incidence of plane waves (for synthetic testing)'''
        azi = np.array([-np.pi/3,-np.pi/8,np.pi/4])
        #azi = np.array([np.pi/4])
        eli = np.zeros_like(azi)
        az,el = self.angles_1d
        faz = scipy.interpolate.interp1d(az,az,'nearest') #find the nearest values
        azi = faz(azi)
        return azi,eli
    
    @property
    def weights_1d(self):
        '''@brief return our weights'''
        return np.ones((self.positions_1d.shape[0]),dtype=np.cdouble)
    
    @property
    def positions_1d(self):
        '''@brief get testing positions'''
        numel = [10,1,1] #number of elements in x,y
        spacing = 2.99e8/np.max(self.freqs)/2 #get our lambda/2 at max freq
        Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
        pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
        return pos
    
    def plot_1d_calc(self):
        '''@brief plot 1D calculated data with dots for correct reference'''
        self._init_plotting()
        go = self.options['plotter']
        myaoa = self.options['aoa_class']()
        az,el = self.angles_1d
        pos = self.positions_1d
        freqs = self.freqs
        meas_vals = self.meas_vals_1d
        out_vals = myaoa.calculate(freqs,pos,meas_vals,az,el)
        #plt.plot(az,10*np.log10(np.abs(out_vals.T)))
        fig = go.Figure()
        scat_list = [fig.add_trace(go.Scatter(
                    x=az,y=10*np.log10(np.abs(ov))
                    ,name=str(freq)+' Hz',mode='lines')) 
                    for ov,freq in zip(out_vals,freqs)]
        azi,_ = self.incident_angles_1d
        azi_mags = np.ones_like(azi)
        #plt.scatter(azi,10*np.log10(np.abs(azi_mags)))
        fig.add_trace(go.Scatter(x=azi,y=10*np.log10(np.abs(azi_mags))
                                 ,name='Correct Values',mode='markers'))
        fig.update_layout(xaxis_title='Angle (radians)',yaxis_title='Magnitude (db)')
        fig.show(renderer='browser')
        return fig
        

            
#%% main tests
if __name__=='__main__':
    unittest.main()
    mya = AoaAlgorithm()
    freqs = np.arange(1,25)
    freqs = 25
    l2 = AoaAlgorithm.SPEED_OF_LIGHT/np.max(freqs)/10
    px = np.arange(-20,20)*l2; py = np.zeros_like(px)*l2; pz = np.zeros_like(px)*l2
    pos = np.array([px,py,pz]).transpose()
    az = np.linspace(-np.pi/2,np.pi/2,10)
    el = np.zeros_like(az)
    kv = mya.get_k_vector_azel(freqs,az,el)
    sv = mya.get_steering_vectors(freqs,pos,az,el)
    sd = mya.synthesize_data(freqs,pos,np.array([-np.pi/6]),np.array([0]),mag=1)
    sd2 = mya.synthesize_data(freqs,pos,-np.pi/8,0,mag=1)
    import matplotlib.pyplot as plt
    plt.plot(np.real(sd.T))
    plt.plot(np.real(sd2.T))
#    suite = unittest.TestSuite([TestAoaAlgorithm])
#    suite.run()
    
        
    


