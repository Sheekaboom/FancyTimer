# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from Modulation import Modulation
from Modulation import generate_gray_code_mapping,generate_root_raised_cosine,lowpass_filter
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import cmath

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
        buffer = 20 #number of symbols to buffer on edge
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
        
        #clock = np.zeros_like(times,dtype=np.int8) #clocking pulse train
        #clock[symbol_start:symbol_end:steps_per_symbol] = 1
        
        clock_sine = np.zeros_like(times)
        clock_times = times[symbol_start:symbol_end]-times[symbol_start]
        clock_phase = np.pi
        clock_sine[symbol_start:symbol_end] = (
            np.sin(2*np.pi*self.options['baud_rate']/2*clock_times+clock_phase)
            #*(-np.cos(2*np.pi*(1/clock_times.max())*clock_times)+1)/2
            )
        #pack into class
        myqam = QAMSignal(data,**self.options) #build the waveform class
        myqam.data_iq = iq_vals
        myqam.times = times
        myqam.baseband_dict['i'] = i_sig
        myqam.baseband_dict['q'] = q_sig
        myqam.clock_sine = clock_sine
        #myqam.clock = clock
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
        #get our data
        #decode the iq points
        bb_dict = self.apply_rrc_filter(qam_signal,False)
        i_decode = bb_dict['i']
        q_decode = bb_dict['q']
        clk = qam_signal.clock.astype(np.bool)
        i_vals = i_decode[clk]
        q_vals = q_decode[clk]
        decoded_iq = i_vals+1j*q_vals
        qam_signal.data_iq = decoded_iq
        return bb_dict
    
    def apply_rrc_filter(self,qam_signal,overwrite=True):
        '''
        @brief apply a root raised cosine filter to the i and q baseband signals
            of the provided QAMSignal
        @param[in] qam_signal - QAMSignal object with an encoded i and q baseband
        @param[in/OPT] overwrite - whether or not to overwrite qam_signal data or just return the signals
        '''
        number_of_periods = 20 #number of periods in both directions the filter extends to
        #generate filter
        symbol_period = 1/qam_signal.options['baud_rate'] #in seconds
        time_step = 1/qam_signal.options['sample_frequency']
        rrc_filt_times = np.arange(-symbol_period*number_of_periods,symbol_period*number_of_periods+time_step,time_step)
        rrc_filt = generate_root_raised_cosine(1,symbol_period,rrc_filt_times)
        rrc_filt = rrc_filt/rrc_filt.max() #normalize peak to 1
        #now apply ot hte baseband signals
        baseband_dict = {}
        for key,val in qam_signal.baseband_dict.items():
            baseband_dict[key] = np.convolve(val,rrc_filt,'same')
        if overwrite:
            qam_signal.baseband_dict = baseband_dict
        return baseband_dict
    
    ##########################################################################
    # Functions for up and downconverting to the carrier frequency
    ##########################################################################
    def upconvert(self,qam_signal):
        '''
        @brief upconvert our baseband signals with a provided carrier frequency
        @param[in] qam_signal - structure for the qam signal with encoded baseband
        '''
        fc = self.options['carrier_frequency']
        II = qam_signal.baseband_dict['i']*np.cos(2.*np.pi*fc*qam_signal.times);
        QQ = qam_signal.baseband_dict['q']*-np.sin(2.*np.pi*fc*qam_signal.times);
        IQ = II+QQ;
        qam_signal.rf_signal = IQ
        
    def downconvert(self,qam_signal):
        '''
        @brief downconvert our rf signal back down to our baseband. 
        '''
        fc = self.options['carrier_frequency']
        i_bb =  qam_signal.rf_signal*np.cos(np.pi*2.*fc*qam_signal.times);#/np.sum(np.square(np.cos(np.pi*2.*self.fc*times)))
        q_bb = -qam_signal.rf_signal*np.sin(np.pi*2.*fc*qam_signal.times);#/np.sum(np.square(np.sin(np.pi*2.*self.fc*times)))
        
        #i_bb = lowpass_filter(i_bb,1/self.options['sample_frequency'],2e9)
        #q_bb = lowpass_filter(q_bb,1/self.options['sample_frequency'],2e9)
        
        #shift clock triggers too
        #clk_sin = qam_signal.clock_sine
        #clk_sin = lowpass_filter(clk_sin,1/self.options['sample_frequency'],300e6)
        
        qam_signal.baseband_dict['i'] = i_bb
        qam_signal.baseband_dict['q'] = q_bb
        #qam_signal.clock_sine = clk_sin
        
        baseband_dict = {'i':i_bb,'q':q_bb}
        return baseband_dict
    
    ##########################################################################
    # Functions for correction of mag/phase
    ##########################################################################
    def calculate_iq_correction(self,input_signal,output_signal):
        '''
        @brief calulcate a magnitude phase offsets for our qam points
        @param[in] output_signal - QAMSignal with incorrect data
        @param[in] input_signal  - QAMSignal with original correct data
        '''
        iq_vals,_ = self.map_to_constellation(data)
        
        #get the true magnitude phase of each symbol
        true_phase = np.angle(input_signal.data_iq)
        true_mag   = np.abs(input_signal.data_iq)
        #now get the magnitudes and phases of our measured
        meas_phase = np.angle(output_signal.data_iq)
        meas_mag   = np.abs(output_signal.data_iq)
        #now get the mean phase and mean scale of the mag
        mag_cor = np.mean(true_mag/meas_mag)
        phase_cor = np.mean(true_phase-meas_phase)
        qam_correction = QAMCorrection(mag_cor,phase_cor)
        return qam_correction
    

    def correct_iq(self,qam_signal,qam_correction):
        '''
        @brief apply calculated correction to current baseband iq signals. if a correction
            doesnt currently exist than calculate it
        @param[in] qam_signal - signal to adjust
        @param[in] qam_correction - QAMCorrection Class to correct the data
        '''
        data = qam_signal.I+1j*qam_signal.Q
        data = qam_correction.correct_iq_data(data)
        qam_signal.I = data.real
        qam_signal.Q = data.imag
        #now decode again
        self.decode_baseband(qam_signal)
        
    def calculate_time_shift(self,input_signal,output_signal):
        '''
        @brief calculate the time shift caused by the channel
            This is currently done using a correlation approach and assuming
            there is a consistent shift in the time
        '''
        center_idx = (output_signal.I.shape[0]/2) #center index. assume I and q same length
        #find the average shift between i and q
        corr_i_idx = np.correlate(input_signal.I,output_signal.I,mode='same').argmax()
        corr_q_idx = np.correlate(input_signal.Q,output_signal.Q,mode='same').argmax()
        corr_mean_idx = np.mean([corr_i_idx,corr_q_idx])
        shift = int(center_idx-corr_mean_idx)
        return shift
    
    def shift_clock(self,qam_signal,shift):
        '''
        @brief adjust the clock of a qam signal by a given number of indices
        @param[in] qam_signal - signal to shift clock of
        @param[in] shift - integer to shift the signal indices by
        '''
        qam_signal.clock_sine = np.roll(qam_signal.clock_sine,shift)

    ##########################################################################
    # Functions for applying things like channels to rf signal
    ##########################################################################
    
    
    ##########################################################################
    # Functions for metrics calculations
    ##########################################################################
    def calculate_evm(self,input_signal,output_signal):
        '''
        @brief calculate the evm from a signal. This is calulcated from equation (1)
            from "Analysis on the Denition Consistency Problem of EVM Measurement 
            and Its Solution" by Z. Feng, J. Riu, S. Jing-Lu, and Z. Xin
        @param[in] input_signal - QAMSignal with the ideal IQ locations in data_iq
        @param[in] output_signal - QAMSignal with the decoded IQ locations in data_iq
        '''
        evm_num = np.abs((output_signal.data_iq-input_signal.data_iq)**2).sum()
        evm_den = np.abs(input_signal.data_iq**2).sum()
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
    def __init__(self,data=None,**arg_options):
        '''
        @brief constructor for the class
        @param[in\OPT] data - data to be modulated
        '''
        self.data_type = type(data)
        self.data = data
        self.data_iq = None
        super().__init__(**arg_options)
        
    @property
    def clock(self):
        '''
        @brief get a clock pulse train from sinusoid zero crossings
        '''
        #normalize values to 1
        #clock_norm = self.clock_sine/self.clock_sine.max()
        #round to prevent small changes from effecting
        #clock_round = np.round(clock_norm,10)
        clk_tf = self.clock_sine>=0
        clk_diff = np.diff(clk_tf)
        clk_diff = np.append(clk_diff,False)
        clk = clk_diff.astype(np.int8)
        return clk
    
    @property
    def I(self):
        '''
        @brief getter for baseband I values
        '''
        return self.baseband_dict['i']
    @I.setter
    def I(self,val):
        '''
        @brief setter for baseband I values
        '''
        self.baseband_dict['i'] = val
    
    @property
    def Q(self):
        '''
        @brief getter for baseband Q values
        '''
        return self.baseband_dict['q']
    @Q.setter
    def Q(self,val):
        '''
        @brief setter for baseband Q values
        '''
        self.baseband_dict['q'] = val
    
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

class QAMCorrection():
    '''
    @brief class to hold and correct IQ data phase/mag
    '''
    def __init__(self,magnitude_correction,phase_correction,**arg_options):
        '''
        @brief constructor
        @param[in] magnitude_correction - multiplier value for magnitude correction
        @param[in] phase_correction - correction for phase in radians
        '''
        self.magnitude_correction = magnitude_correction
        self.phase_correction = phase_correction
        
    def correct_iq_data(self,data):
        '''
        @brief correct an array of complex iq data with the current mag and phase corrections
        @param[in] data - numpy array of complex data
        '''
        data = self._correct_iq_mag(data)
        data = self._correct_iq_phase(data)
        return data
        
    def _correct_iq_mag(self,data):
        '''
        @brief apply magnitude correction to complex data
        '''
        corrected_data = np.zeros_like(data)
        for i,val in enumerate(data):
           mag,phase = cmath.polar(val)
           mag *= self.magnitude_correction
           adj_iq = cmath.rect(mag,phase)
           corrected_data[i] = adj_iq
        return corrected_data
           
    def _correct_iq_phase(self,data):
        '''
        @brief apply plahse correction to complex data
        '''
        corrected_data = np.zeros_like(data)
        for i,val in enumerate(data):
           mag,phase = cmath.polar(val)
           phase += self.phase_correction
           adj_iq = cmath.rect(mag,phase)
           corrected_data[i] = adj_iq
        return corrected_data

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


import copy
if __name__=='__main__':
    
    #codes = generate_gray_code_mapping(32)
    #for c in np.flip(codes,axis=1):
    #    print(c)

    mymodem = QAMModem(16)
    #fig=myqm.plot_constellation()
    #mymap,inbits = myq.map_to_constellation(bytearray('testing'.encode()))
    data = 'testing'.encode()
    #data = np.random.random(5)
    #data = [255,255,255]
    
    print("Encoding")
    inqam = mymodem.encode_baseband(bytearray(data))
    mymodem.upconvert(inqam)
    #myqam.plot_baseband()
    #myqam.plot_rf()
    
    #upconvert
    print("Upconverting")
    mymodem.upconvert(inqam)
    
    #run through channel
    print("Applying Channel")
    outqam = copy.deepcopy(inqam)

    #downconvert
    print("Downconverting")
    mymodem.downconvert(outqam)
    
    #decode the data
    print("Decoding")
    mymodem.decode_baseband(outqam)
    
    #correct the data
    print("Applying Correction")
    #time correction
    time_shift = mymodem.calculate_time_shift(inqam,outqam)
    mymodem.shift_clock(outqam,time_shift)
    #mag/phase correction
    correction = mymodem.calculate_iq_correction(inqam,outqam)
    mymodem.correct_iq(outqam,correction)
    
    #calculate metrics
    print("Calculating EVM")
    evm = mymodem.calculate_evm(inqam,outqam)
    print(evm)

    
    
    
    
    

    
    

    
   
   
   
   
   
    