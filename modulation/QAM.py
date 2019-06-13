# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from Modulation import Modulation
from Modulation import generate_gray_code_mapping,generate_root_raised_cosine
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

class QAMModem(Modulation):
    '''
    A class to modulate a set of data using an M-QAM modulation scheme.
    Inherits from the Modulation superclass
    
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
    module_level_variable1 : int
        Module level variables may be documented in either the ``Attributes``
    
    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    '''
    def __init__(self,M,**arg_options):
        '''
        @brief initialize the modulation class
        @param[in/OPT] M - type of QAM (e.g 4,16,256)
        '''
        defaults = {} #default options
        arg_options.update(defaults)
        super().__init__(**arg_options)
        self.M = M #what qam we are using
        self.constellation_dict = {} #dictionary containing key/values for binary/RI constellation
        self._generate_mqam_constellation(M)
        
    ##########################################################################
    # functions for time domain baseband decoding/encoding
    ##########################################################################
    
    def encode_baseband(self,data,**arg_options):
        '''
        @brief encode a set of data into a baseband time domain waveform
        @param[in] data - a set of data to encode (if is a bitstream set bitstream=True)
        @param[in/OPT] arg_options - keyword options as follows:
            None for this function yet
            Passed to self.map_to_constellation
        '''
        iq_vals,_ = self.map_to_constellation(data,**arg_options) #get our complex data
        i_vals = iq_vals.real; q_vals = iq_vals.imag
        #generate time information
        buffer = 2 #number of symbols to buffer on edge
        num_symbols = len(iq_vals)+buffer*2
        symbol_period = 1/self.options['baud_rate'] #in seconds
        time_step = 1/self.options['sample_frequency']
        steps_per_symbol = symbol_period/time_step
        symbol_start = int(steps_per_symbol*(0.5+buffer)); #0.5+buffer for buffer
        symbol_end   = int(symbol_start+steps_per_symbol*len(iq_vals))
        steps_per_symbol = int(steps_per_symbol)
        times = np.arange(0,symbol_period*num_symbols+time_step,time_step)
        #now generate our iq pulse train
        i_sig = np.zeros_like(times); q_sig = np.zeros_like(times)
        i_sig[symbol_start:symbol_end:steps_per_symbol] = i_vals
        q_sig[symbol_start:symbol_end:steps_per_symbol] = q_vals
        clock = np.zeros_like(times,dtype=np.int8) #clocking pulse train
        clock[symbol_start+1:symbol_end:steps_per_symbol] = 1
        #pack into class
        myqam = QAMSignal(data,**self.options) #build the waveform class
        myqam.data_iq = iq_vals
        myqam.times = times
        myqam.i_baseband = i_sig
        myqam.q_baseband = q_sig
        myqam.clock = clock
        #now filter
        self.apply_rrc_filter(myqam)

        return myqam
    
    def decode_baseband(self,qam_signal):
        '''
        @brief decode a time domain baseband waveform to complex iq points. This
            data will be placed into qam_signal.decoded_iq
        @param[in] qam_signal - QAMSignal object with an encoded i and q baseband
            and a clock pulse train
        '''
        #decode the iq points
        clk = qam_signal.clock.astype(np.bool)
        i_vals = qam_signal.i_baseband[clk]
        q_vals = qam_signal.q_baseband[clk]
        decoded_iq = i_vals+1j*q_vals
        qam_signal.decoded_iq = decoded_iq
        #now unmap to find the returned data (just for fun pretty much)
        #this will be a bitstream
        #_,bitstream = self.unmap_from_constellation(decoded_iq)
        #qam_signal.decoded_bitstream = bitstream
    
    def apply_rrc_filter(self,qam_signal):
        '''
        @brief apply a root raised cosine filter to the i and q baseband signals
            of the provided QAMSignal
        @param[in] qam_signal - QAMSignal object with an encoded i and q baseband
        '''
        #generate filter
        symbol_period = 1/qam_signal.options['baud_rate'] #in seconds
        time_step = 1/qam_signal.options['sample_frequency']
        rrc_filt_times = np.arange(-symbol_period/2,symbol_period/2+time_step,time_step)
        rrc_filt = generate_root_raised_cosine(1,symbol_period,rrc_filt_times)
        rrc_filt = rrc_filt/rrc_filt.max() #normalize peak to 1
        #now apply ot hte baseband signals
        qam_signal.i_baseband = np.convolve(qam_signal.i_baseband,rrc_filt,'same')
        qam_signal.q_baseband = np.convolve(qam_signal.q_baseband,rrc_filt,'same')
    
    ##########################################################################
    # Functions for correction of mag/phase
    ##########################################################################
    def calculate_iq_correction(self,qam_signal):
        '''
        @brief calulcate a magnitude phase offsets for our qam points
        '''
        pass
    
    
    ##########################################################################
    # Functions for metrics calculations
    ##########################################################################
    def calculate_evm(self,qam_signal):
        '''
        @brief calculate the evm from a signal. This is calulcated from equation (1)
            from "Analysis on the Denition Consistency Problem of EVM Measurement 
            and Its Solution" by Z. Feng, J. Riu, S. Jing-Lu, and Z. Xin
        @param[in] qam_signal - QAMSignal class with a decoded_iq property and
            a data_iq property
        '''
        evm_num = np.abs((qam_signal.decoded_iq-qam_signal.data_iq)**2).sum()
        evm_den = np.abs(qam_signal.data_iq**2).sum()
        evm = np.sqrt(evm_num/evm_den)*100
        return evm
        
    
    ##########################################################################
    # Functions for dealing with the constellation
    ##########################################################################
    
    def _generate_mqam_constellation(self,M):
        '''
        @brief generate an M-QAM constellation
        @param[in] M - order of QAM (must be power of 2)
        '''
        self.constellation_dict = generate_gray_code_mapping(M,generate_qam_position)
        
    def plot_constellation(self,**arg_options):
        '''
        @brief plot the constellation points defined in constellation_dict
        '''
        #configure the figure
        fig = plt.figure()
        ax  = plt.axes()
        ax.set_xlim([-1.5,1.5])
        ax.set_ylim([-1.5,1.5])
        #ax.grid(linestyle='dotted')
        for k,v in self.constellation_dict.items():
            plt.plot(v.real,v.imag,'bo')
            plt.text(v.real,v.imag+0.05,"".join(str(int(x)) for x in k),ha='center')
        return fig
    
    def map_to_constellation(self,data,**arg_options):
        '''
        @brief map input data to constellation. If the number of bits does
        not fit in the encoding then pad with the value given by padding
        @param[in] data - data to map. currently supports 'uint'
            for raw bitstream use named argument 'bitstream' to True
        @param[in/OPT] arg_options - optional keyword arguments as follows:
            padding - value to pad bits with if encoding doesnt fit (default 0)
            bitstream - True if the data is a bitstream (array of 1s and 0s) (default False)
        @return a list of complex numbers for the corresponding mapping, mapped bitstream
        '''
        options = {}
        options['padding'] = 0
        options['bitstream'] = False
        for key,val in arg_options.items():
            options[key] = val
        data = bytearray(data) #to bytearray
        bitstream = np.unpackbits(data) #to bits
        mlog2 = np.log2(self.M)
        mlog2 = int(round(mlog2)) #maybe deal with this in the future
        len_mod = len(bitstream)%mlog2
        if len_mod is not 0: #extra bits we need
            print("Warning bits not divisible by encoding (%d/%d). Padding end with %ds" %(len(bitstream),mlog2,options['padding']))
            extra_bits = round(mlog2-len_mod)
            bitstream = np.append(bitstream,np.full((extra_bits,),options['padding']))
        split_bits = np.split(bitstream,len(bitstream)/mlog2) #split into packets
        locations = []
        for pack in split_bits:
            loc = self.get_constellation_location(pack)
            locations.append(loc)
        return np.array(locations),bitstream
    
    def unmap_from_constellation(self,locations):
        '''
        @brief unmap a set of iq (complex) values from the constellation
        @param[in] locations - iq values to unmap
        @return a bytearray of the unmapped values, array of bits (bitstream)
        '''
        vals = self.get_constellation_values(locations)
        bitstream = vals.reshape((-1,)).astype('int')
        packed_vals = bytearray(np.packbits(bitstream))
        return packed_vals,bitstream
    
    def get_constellation_location(self,bits):
        '''
        @brief get a location (complex number) on the constellation given
            a set of bits. These bits should be an array (or list) of ones and
            zeros in LSB first order (how np.unpackbits provides values)
        @param[in] bits - list (or array) of bits in LSB first order (usually form np.unpackbits)
        '''
        #some checks
        mlog2 = np.log2(self.M)
        if len(bits) != mlog2:
            raise ValueError("Number of bits must equal Log2(M)")
        if not np.all(np.logical_or(bits==1,bits==0)):
            raise ValueError("Bits argument must be a list (or array) of only zeros or ones")
        key = tuple(np.flip(bits,axis=0)) #MSB first for key
        val = self.constellation_dict[key]
        return val
    
    def get_constellation_values(self,locations):
        '''
        @brief get a value (code) from a given constellation location(s) (complex number)
        @param[in] locations - complex number(s) for constellation value
        '''
        if not hasattr(locations,'__iter__'):
            locations = [locations]
        location_dict = self.location_constellation_dict
        values = []
        for l in locations:
            val = np.flip(location_dict[l],axis=0)
            values.append(val)
        return np.array(values)
    
    @property
    def codes(self):
        '''
        @brief getter for the coding of the qam
        '''
        codes = []
        for k,v in self.constellation_dict.items():
            codes.append(k)
        return np.array(codes)
    
    @property
    def locations(self):
        '''
        @brief getter for the constellation locations (complex numbers)
        '''
        locations = []
        for k,v in self.constellation_dict.items():
            locations.append(v)
        return np.array(locations)
    
    @property
    def location_constellation_dict(self):
        '''
        @brief this returns a dictionary the same as constellation_dict but with
        the key/value pairs flipped
        '''
        location_map_dict = {}
        for k,v in self.constellation_dict.items():
            location_map_dict[v] = k
        return location_map_dict
   
   #def modulate()
   #    pas 
   
from Modulation import ModulatedSignal
class QAMSignal(ModulatedSignal):
    '''
    @brief class to hold data for a qam modulated signal. This works alongisde the QAM class
    '''
    def __init__(self,data,**arg_options):
        '''
        @brief constructor for the class
        @param[in] data - data to be modulated
        '''
        self.data_type = type(data)
        self.data = data
        self.data_iq = None
        super().__init__(**arg_options)
        self.i_baseband = None
        self.q_baseband = None
        self.times      = None
        self.clock      = None #pulse train for the sampling locations
        
        #decoded values
        self.decoded_iq = None
        self.decoded_bitstream = None
        self.iq_correction = None
        
    def plot_baseband(self):
        '''
        @brief plot the i and q baseband signals
        '''
        fig = plt.figure()
        plt.plot(self.times,self.i_baseband,label="I Baseband")
        plt.plot(self.times,self.q_baseband,label="Q Baseband")
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')
        return fig
    
    def plot_iq(self,ax=None):
        '''
        @brief plot the current i_baseband and q_baseband onto a 2d iq plot
        @param[in/OPT] ax - axis to plot on. if none make one
        '''
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(self.i_baseband,self.q_baseband)
        return ax
    
    def plot_decoded_points(self,ax=None):
        '''
        @brief plot the decoded data onto an iq plot (constellation diagram)
        @param[in/OPT] ax - axis to plot on. if none make one
        '''
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.scatter(self.decoded_iq.real,self.decoded_iq.imag)

def generate_qam_position(code_number,num_codes):
    '''
    @brief generate a qam position based on a given code number and total number of codes
    This is typically passed to generate_gray_code_mapping()
    This currently only works for a power of 2
    '''
    row_len = round(np.sqrt(num_codes)) #get the length of the side (e.g. 4 for 16 QAM)
    col_len = row_len
    #normalize size from -1 to 1
    step_size = 1/(((row_len-1)/2))
    imag_part = (code_number%row_len)*step_size-1
    real_part = (np.floor(code_number/row_len)*step_size)-1
    return complex(real_part,imag_part)



if __name__=='__main__':
    
    #codes = generate_gray_code_mapping(32)
    #for c in np.flip(codes,axis=1):
    #    print(c)
    myqm = QAMModem(64)
    fig=myq.plot_constellation()
    #mymap,inbits = myq.map_to_constellation(bytearray('testing'.encode()))
    #data = 'testing'.encode()
    data = np.random.random(20)
    myqam = myqm.encode_baseband(bytearray(data))
    #myqam.plot_baseband()
    #plt.plot(myqam.times,myqam.clock*2-1)
    myqm.apply_rrc_filter(myqam)
    myqam.i_baseband = myqam.i_baseband/np.abs(myqam.i_baseband).max()
    myqam.q_baseband = myqam.q_baseband/np.abs(myqam.q_baseband).max()
    myqam.plot_baseband()
    #plt.plot(myqam.times,myqam.clock*2-1)
    myqm.decode_baseband(myqam)
    ax = fig.gca()
    myqam.plot_iq(ax)
    myqam.plot_decoded_points(ax)
    myevm = myqm.calculate_evm(myqam)
    print(myevm)
    #inbits = np.unpackbits(bytearray('testing'.encode()))
    #plt.plot(mymap.real,mymap.imag-0.01)
    #vals,outbits = myq.unmap_from_constellation(mymap)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
   
   
   
    