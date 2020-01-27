# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:28:05 2019
Classes are labeled with numpy style, meethods are labeled with doxygen style
@author: aweis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import unittest

from pycom.modulation.QAM import QAMConstellation
try:
    from samurai.base.SamuraiDict import SamuraiDict #this has some nice extra features
except ModuleNotFoundError:
    from collections import OrderedDict as SamuraiDict #this is pretty close

class Modem(SamuraiDict):
    '''
    @brief constructor to get options (and do other things)
    @param[in/OPT] arg_options - keyword arguments as follows
        - sample_frequency - frequency for sample points
        - carrier_frequency - frequency our qam will be upconverted to upon modulation
   '''
    def __init__(self,M,**arg_options):
        '''
        @brief constructor to get options (and do other things)
        @param[in] M - value specifying the modulation (e.g. M-QAM like 16 or 'BPSK','16QAM')
        @param[in/OPT] arg_options - keyword arguments as follows
            sample_frequency - frequency for sample points
            carrier_frequency - frequency our qam will be upconverted to upon modulation
        '''
        super().__init__()
        self['constellation'] = QAMConstellation(M,**arg_options)
        for k,v in arg_options.items():
            self[k] = v
   
    def modulate(self,input_data):
        '''
        @brief Map data to an iq constellation.vThis uses self['constellation'] to perform mapping
        @param[in] input_data - data to map. 
        @return modulated complex values
        '''
        return self['constellation'].map(input_data)
       
    def demodulate(self,iq_data,dtype):
        '''
        @brief Unmap data from an iq constellation. this uses the self['constellation'] values
        @param[in] input_data - data to map. This uses self['constellation'] to perform mapping
        @return modulated complex values
        '''
        data = self['constellation'].unmap(iq_data,dtype)
        return data[0] #do not return error here (although maybe we should?)
       
    def upconvert(self):
        raise NotImplementedError("Please implmement a 'upconvert' method")
       
    def downconvert(self):
        raise NotImplementedError("Please implmenet a 'downconvert' method")

    @property
    def options(self):
        '''
        @brief have this to be backward compatable with options dictionary
        '''
        return self

    def __getattr__(self,name):
        '''
        @brief check our options dictionary if an attirbute doesnt exist
        '''
        try: #return value from the dictionary
            rv = self[name]
            return rv
        except KeyError:
            raise AttributeError("{} is not an attribute or key of {}",name,type(self))

    
def generate_gray_code_mapping(num_codes,constellation_function):
    '''
    @brief generate gray codes for a given number of codes
    @param[in] num_codes - the number of codes to generate (e.g. 256 is 0-255)
    @param[in] constellation_function - function to generate constellation locations
                for each code. The input to this will be the number of codes and the current code number.
                the return should be a complex number of where the point is on the real imaginary plane
    @return a dictionary mapping an array of bits to a constellation locations
    '''
    num_bits = get_num_bits(num_codes-1)
    codes = []
    constellation_dict = {}
    for i in range(num_codes): #for each desired code
        cur_const = constellation_function(i,num_codes)#generate our locations for iq here
        cur_code = np.zeros(num_bits) #flip to get MSB first in array
        for bn in range(num_bits):
            cur_code[bn] = (((i+2**bn)>>(bn+1)))%2
        cur_code = np.flip(cur_code,axis=0)
        codes.append(cur_code)
        constellation_dict[tuple(cur_code)] = cur_const
    return constellation_dict
   
def get_num_bits(number):
    '''
    @brief find the number of bits required for a number
    '''
    num_bits = np.ceil(np.log2(number)).astype('int') #much easier way
    #myba   = bytearray(np.array(number).byteswap().tobytes())
    #mybits = np.unpackbits(myba)
    #ones_loc = np.where(mybits==1)[0] #location of ones
    #if ones_loc.shape[0] == 0: #if the list is empty
    #    return 0 #if the number is 0 we have 0 bits
    #num_bits = mybits.shape[0] - ones_loc[0]
    return num_bits
   
def generate_root_raised_cosine_v1(beta,Ts,times):
    '''
    @brief generate a raised root cosine filter (from wikipedia equations)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    h = np.zeros(times.shape)
    for i,t in enumerate(times): #go through each time and calculate
        if t is 0:
            h[i] = 1/Ts*(1+beta(4/np.pi - 1))
        elif t is Ts/(4*beta) or t is -Ts/(4*beta):
            h[i] = beta/(Ts*np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            h[i] = 1/Ts*(np.sin(np.pi*(t/Ts)*(1-beta))+4*beta*(t/Ts)*np.cos(np.pi*(t/Ts)*(1+beta)))/(np.pi*(t/Ts)*(1-(4*beta*(t/Ts))**2))
    return h
    #right now runs saved values in
    #def run_impulse_response()
    
def generate_raised_cosine(beta,Ts,times):
    '''
    @brief generate a raised cosine filter (from  https://dspguru.com/dsp/reference/raised-cosine-and-root-raised-cosine-formulas/)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    sin_term = np.sin(np.pi*times/Ts)/(np.pi*times)
    cos_term = np.cos(np.pi*times*beta/Ts)/(1-4*beta**2*times**2/Ts**2)
    h = sin_term*cos_term
    return h
    #right now runs saved values in
    #def run_impulse_response()
    
def generate_root_raised_cosine(beta,Ts,times):
    '''
    @brief generate a raised root cosine filter (from  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120008631.pdf)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    term_1 = 2*beta/(np.pi*np.sqrt(Ts))
    term_2_numerator = np.cos((1+beta)*np.pi*times/Ts)+np.sin((1-beta)*np.pi*times/Ts)/(4*beta*times/Ts)
    term_2_denominator = 1-(4*beta*times/Ts)**2
    h = term_1*term_2_numerator/term_2_denominator
    return h
    #right now runs saved values in
    #def run_impulse_response()


import scipy.signal
def lowpass_filter(data,time_step,cutoff_freq,order=5):
        # from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
        nyq = 0.5/time_step;
        norm_cut = cutoff_freq/nyq;
        b,a = scipy.signal.butter(order,norm_cut,btype='low',analog=False)
        out = scipy.signal.lfilter(b,a,data);
        return out;
    
def lowpass_filter_zero_phase(data,time_step,cutoff_freq,order=5):
        # from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    nyq = 0.5/time_step;
    norm_cut = cutoff_freq/nyq;
    b,a = scipy.signal.butter(order,norm_cut,btype='low',analog=False)
    out = scipy.signal.filtfilt(b,a,data);
    return out;

def bandstop_filter(data,time_step,cutoff_freq,q_factor=20):
    nyq = 0.5/time_step;
    norm_cut = cutoff_freq/nyq;
    b,a = scipy.signal.iirnotch(norm_cut,q_factor)
    out = scipy.signal.filtfilt(b,a,data);
    return out;

def plot_frequency_domain(data,dt):
    f_data = np.fft.fft(data)
    f_freqs= np.fft.fftfreq(len(f_data),dt)
    fig = plt.figure()
    plt.plot(f_freqs,f_data)
    return fig

#naive fourier series calculations (non-uniform foureier transform)
def dft(data,times,freqs_out):
    '''
    @brief calculate the fouerier series for nonuniform points
    @param[in] data - f(t) values for the corresponding times
    @param[in] times - times that correspond to the data points
    @param[in] freqs_out - list of output frequencies to calculate for
    '''
    #ran out of memory for vectorized computation
    freq_vals = []
    print("%5d of %5d" %(0,len(freqs_out)),end='')
    for i,f in enumerate(freqs_out):
        print('\b'*14+"%5d of %5d" %(i+1,len(freqs_out)),end='')
        data  = np.array(data).flatten()
        times = np.array(times).flatten()
        exp_val = np.exp(-1j*2*np.pi*f*times)
        fv = (data*exp_val).sum()
        freq_vals.append(fv)
    print('')
    return np.array(freq_vals)

def idft(data,freqs,times_out):
    '''
    @brief calculate the inverser fourier series at nonuniform points
    @param[in] data - F(f) values at corresponding frequencies
    @param[in] freqs - corresponding frequencies to data
    @param[in] times_out - times to calculate for
    '''
    times_out = -times_out #change -1j to 1j by dong this
    time_vals = dft(data,freqs,times_out)
    return time_vals

def complex2magphase(data):
    return np.abs(data),np.angle(data)

def magphase2complex(mag,phase):
    real = mag*np.cos(phase)
    imag = mag*np.sin(phase)
    return real+1j*imag

class TestModem(unittest.TestCase):
    '''@some tests for the modem'''
    def test_mod_demod(self):
        '''@brief test modulation demodulation of data to qam constellations'''
        '''@brief test mapping/unmapping to constellations'''
        m_vals = ['bpsk','qpsk','16qam']
        bits_per_symbol = [1,2,4]
        data_len = 1000
        mydtype = np.float16
        bytes_per_data = len(np.zeros((1),dtype=mydtype).tobytes())
        data_in = np.random.rand(data_len).astype(mydtype)
        for m,bps in zip(m_vals,bits_per_symbol):
            mymodem = Modem(m)
            iq_data  = mymodem.modulate(data_in)
            self.assertEqual(len(iq_data), (bytes_per_data*data_len*8)/bps)#ensure correct length
            data_out = mymodem.demodulate(iq_data,mydtype)
            self.assertTrue(np.all(data_in==data_out),msg='Failed on {}'.format(m))

if __name__=='__main__':
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModem)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    #import os
    #import numpy as np
    #modsig = ModulatedSignal(np.arange(10),np.arange(10)+1j*np.arange(10,20))
    #modsig.metadata['new_data'] = 'this is some new data'
    #modsig.metadata['data_list'] = [1,2,3,4,5]
    #modsig.write('test.modsig')
    #modsig2 = ModulatedSignal('test.modsig')
    #os.remove('test.modsig')
    
    
#    import matplotlib.pyplot as plt
#    t = np.arange(-10,11,0.1)
#    ts = 1
#    betas = [0.001,0.01,0.1,0.5,0.7,1]
#    fig = plt.figure()
#    for b in betas:
#        h = generate_root_raised_cosine(b,ts,t)
#        plt.plot(t,h,label=r"$\beta={}$".format(b))
#    ax = plt.gca()
#    ax.legend()
#    
#    times = np.arange(0,100,0.1)
#    dirac = np.zeros(times.shape)
#    dirac[int(times.shape[0]/2)] = 1
#    plt.figure()
#    plt.plot(times,dirac)
#    dirac_conv = np.convolve(dirac,h,'same')
#    plt.plot(times,dirac_conv)
#    dirac_conv_conv = np.convolve(dirac_conv,h,'same')
#    plt.plot(times,dirac_conv_conv)  














