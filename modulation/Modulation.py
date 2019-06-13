# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:28:05 2019
Classes are labeled with numpy style, meethods are labeled with doxygen style
@author: aweis
"""

import numpy as np

class Modulation:
    '''
    This is a class for describing a modulation scheme. This will be inherited 
    by other modulation modules as subclasses.
    
    Example
    -------
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::
    
        $ python example_numpy.py
    
    
    Section breaks are created with two blank lines. Section breaks are also
    implicitly created anytime a new section starts. Section bodies *may* be
    indented:
    
    Notes
    -----
        This is an example of an indented section. It's like any other section,
        but the body is indented to help it stand out from surrounding text.
    
    If a section is indented, then a section break is created by
    resuming unindented text.
    
    Attributes
    ----------
    modulated_data
        numpy list of time domain modulated data
    
    Methods
    -------
    modulate(data)
        Overwrite method to modulate a set of input data.
        data input should be be a bytearray
        Raise NotImplementedError if not overwritten.
   '''
    def __init__(self,**arg_options):
        '''
        @brief constructor to get options (and do other things)
        @param[in/OPT] arg_options - keyword arguments as follows
            sample_frequency - frequency for sample points
            carrier_frequency - frequency our qam will be upconverted to upon modulation
        '''
        self.options = {}
        self.options['sample_frequency']   = 100e9
        self.options['carrier_frequency'] = 1e9
        self.options['baud_rate']         = 100e6
        for k,v in arg_options.items():
            self.options[k] = v
        
   
    def modulate(self,data):
        '''
        @brief method to overwrite to modulate the data
        @param[in] data - input data to modulate
        '''
        raise NotImplementedError("Please implement a 'modulate' method")
       
    def demodulate(self):
        raise NotImplementedError("Please implmenet a 'demodulate' method")
       
    def upconvert(self):
        raise NotImplementedError("Please implmement a 'upconvert' method")
       
    def downconvert(self):
        raise NotImplementedError("Please implmenet a 'downconvert' method")

       
    def get_bits(self,data):
        '''
        @brief method to get the bits from input data
        @param[in] data - input data to get bits from
        '''
        data_ba = bytearray(data)
        bits = np.unpackbits(data_ba)
        return bits
    
class ModulatedSignal:
    '''
    @brief class to provide a template for a modulated signal
    This is simply a structure to hold all of the data describing the signal
    '''
    def __init__(self,**arg_options):
        '''
        @brief constructor to get options (and do other things)
        @param[in/OPT] arg_options - keyword arguments as follows
            sample_frequency - frequency for sample points
            carrier_frequency - frequency our qam will be upconverted to upon modulation
        '''
        self.options = {}
        self.options['sample_frequency']   = 100e9
        self.options['carrier_frequency'] = 1e9
        self.options['baud_rate']         = 1e6
        self.options['type']              = None
        for k,v in arg_options.items():
            self.options[k] = v
   
   
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
   
def generate_root_raised_cosine(beta,Ts,times):
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
    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    t = np.arange(-10,11,0.1)
    ts = 1
    betas = [0.001,0.01,0.1,0.5,0.7,1]
    fig = plt.figure()
    for b in betas:
        h = generate_root_raised_cosine(b,ts,t)
        plt.plot(t,h,label=r"$\beta={}$".format(b))
    ax = plt.gca()
    ax.legend()
    
    times = np.arange(0,100,0.1)
    dirac = np.zeros(times.shape)
    dirac[int(times.shape[0]/2)] = 1
    plt.figure()
    plt.plot(times,dirac)
    dirac_conv = np.convolve(dirac,h,'same')
    plt.plot(times,dirac_conv)
    dirac_conv_conv = np.convolve(dirac_conv,h,'same')
    plt.plot(times,dirac_conv_conv)  


