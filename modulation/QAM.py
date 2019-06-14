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
        buffer = 15 #number of symbols to buffer on edge
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
        clock_sine[symbol_start:symbol_end] = np.sin(2*np.pi*self.options['baud_rate']/2*clock_times+clock_phase)
        #pack into class
        myqam = QAMSignal(data,**self.options) #build the waveform class
        myqam.data_iq = iq_vals
        myqam.times = times
        myqam.i_baseband = i_sig
        myqam.q_baseband = q_sig
        myqam.clock_sine = clock_sine
        #now filter
        self.apply_rrc_filter(myqam)

        return myqam
    
    def decode_baseband(self,qam_signal,source='if'):
        '''
        @brief decode a time domain baseband waveform to complex iq points. This
            data will be placed into qam_signal.decoded_iq
        @param[in] qam_signal - QAMSignal object with an encoded i and q baseband
            and a clock pulse train
        @param[in/OPT] source - source of the data to decode can be the following:
            rf - decode from downconverted rf signal
            in - decode from baseband that has only been encoded
        '''
        #get our data
        #decode the iq points
        i_decode,q_decode = self.apply_rrc_filter(qam_signal,False)
        clk = qam_signal.clock.astype(np.bool)
        i_vals = i_decode[clk]
        q_vals = q_decode[clk]
        decoded_iq = i_vals+1j*q_vals
        qam_signal.decoded_iq = decoded_iq
        qam_signal._decoded_i_baseband = i_decode
        qam_signal._decoded_q_baseband = q_decode
        #now unmap to find the returned data (just for fun pretty much)
        #this will be a bitstream
        #_,bitstream = self.unmap_from_constellation(decoded_iq)
        #qam_signal.decoded_bitstream = bitstream
    
    def apply_rrc_filter(self,qam_signal,overwrite=True,source='):
        '''
        @brief apply a root raised cosine filter to the i and q baseband signals
            of the provided QAMSignal
        @param[in] qam_signal - QAMSignal object with an encoded i and q baseband
        @param[in/OPT] overwrite - whether or not to overwrite qam_signal data or just return the signals
        '''
        number_of_periods = 15 #number of periods in both directions the filter extends to
        #generate filter
        symbol_period = 1/qam_signal.options['baud_rate'] #in seconds
        time_step = 1/qam_signal.options['sample_frequency']
        rrc_filt_times = np.arange(-symbol_period*number_of_periods,symbol_period*number_of_periods+time_step,time_step)
        rrc_filt = generate_root_raised_cosine(1,symbol_period,rrc_filt_times)
        rrc_filt = rrc_filt/rrc_filt.max() #normalize peak to 1
        #now apply ot hte baseband signals
        i_bb = np.convolve(qam_signal.i_baseband,rrc_filt,'same')
        q_bb = np.convolve(qam_signal.q_baseband,rrc_filt,'same')
        if overwrite:
            qam_signal.i_baseband = i_bb
            qam_signal.q_baseband = q_bb
        return i_bb,q_bb
    
    ##########################################################################
    # Functions for up and downconverting to the carrier frequency
    ##########################################################################
    def upconvert(self,qam_signal):
        '''
        @brief upconvert our baseband signals with a provided carrier frequency
        @param[in] qam_signal - structure for the qam signal with encoded baseband
        '''
        fc = self.options['carrier_frequency']
        II = qam_signal.i_baseband*np.cos(2.*np.pi*fc*qam_signal.times);
        QQ = qam_signal.q_baseband*-np.sin(2.*np.pi*fc*qam_signal.times);
        IQ = II+QQ;
        qam_signal.rf_signal = IQ
        
    def downconvert(self,qam_signal):
        '''
        @brief downconvert our rf signal back down to our baseband and store in 
            qam_signal.if_i and qam_signal.if_q
        '''
        fc = self.options['carrier_frequency']
        i_bb =  qam_signal.rf_signal*np.cos(np.pi*2.*fc*qam_signal.times);#/np.sum(np.square(np.cos(np.pi*2.*self.fc*times)))
        q_bb = -qam_signal.rf_signal*np.sin(np.pi*2.*fc*qam_signal.times);#/np.sum(np.square(np.sin(np.pi*2.*self.fc*times)))
        
        i_bb = lowpass_filter(i_bb,1/self.options['sample_frequency'],300e6)
        
        qam_signal.if_i = i_bb
        qam_signal.if_q = q_bb
        
        return i_bb,q_bb;
    
    ##########################################################################
    # Functions for correction of mag/phase
    ##########################################################################
    def _calculate_iq_correction(self,qam_signal):
        '''
        @brief calulcate a magnitude phase offsets for our qam points
        '''
        #get the true magnitude phase of each symbol
        true_phase = np.angle(qam_signal.data_iq)
        true_mag   = np.abs(qam_signal.data_iq)
        #now get the magnitudes and phases of our measured
        meas_phase = np.angle(qam_signal.decoded_iq)
        meas_mag   = np.abs(qam_signal.decoded_iq)
        #now get the mean phase and mean scale of the mag
        mag_cor = np.mean(true_mag/meas_mag)
        phase_cor = np.mean(true_phase-meas_phase)
        qam_signal.mag_correction = mag_cor
        qam_signal.phase_correction = phase_cor
        return mag_cor,phase_cor
    
    def correct_decoded_iq(self,qam_signal):
        '''
        @brief apply calculated correction to current decoded iq signals. if a correction
            doesnt currently exist than calculate it
        '''
        if qam_signal.mag_correction is None or qam_signal.phase_correction is None: #then calculate the correction
            mag_cor,phase_cor = self._calculate_iq_correction(qam_signal)
        else:
            mag_cor = qam_signal.mag_correction
            phase_cor = qam_signal.phase_correction
        self._adjust_decoded_mag(qam_signal,qam_signal.mag_correction)
        self._adjust_decoded_phase(qam_signal,qam_signal.phase_correction)
        return mag_cor,phase_cor
        
    def correct_iq(self,qam_signal):
        '''
        @brief apply calculated correction to current baseband iq signals. if a correction
            doesnt currently exist than calculate it
        '''
        if qam_signal.mag_correction is None or qam_signal.phase_correction is None: #then calculate the correction
            self._calculate_iq_correction(qam_signal)
        self._adjust_iq_mag(qam_signal,qam_signal.mag_correction)
        self._adjust_iq_phase(qam_signal,qam_signal.phase_correction)
        self.correct_decoded_iq(qam_signal) #also correct the decoded data
    
    def _adjust_iq_mag(self,qam_signal,mag_mult):
        '''
        @brief adjust magnitude of i and q baseband signals
        @param[in] qam_signal - signal to adjust
        @param[in] mag_mult- multiplier to adjust by
        '''
        adjusted_i = np.zeros_like(qam_signal._decoded_i_baseband)
        adjusted_q = np.zeros_like(qam_signal._decoded_q_baseband)
        baseband = qam_signal.i_baseband+1j*qam_signal.q_baseband
        for i,val in enumerate(baseband):
           mag,phase = cmath.polar(val)
           mag *= mag_mult
           adj_iq = cmath.rect(mag,phase)
           adjusted_i[i] = adj_iq.real
           adjusted_q[i] = adj_iq.imag
        qam_signal._decoded_i_baseband = adjusted_i
        qam_signal._decoded_q_baseband = adjusted_q
        
    def _adjust_decoded_mag(self,qam_signal,mag_mult):
        '''
        @brief adjust magnitude of decoded signals
        @param[in] qam_signal - signal to adjust
        @param[in] mag_mult- multiplier to adjust by
        '''
        adjusted_iq = np.zeros_like(qam_signal.decoded_iq,dtype=np.complex128)
        for i,val in enumerate(qam_signal.decoded_iq):
            mag,phase = cmath.polar(val)
            mag *= mag_mult
            adjusted_iq[i] = cmath.rect(mag,phase)
        qam_signal.decoded_iq = adjusted_iq
    
    def _adjust_iq_phase(self,qam_signal,phase_rot_rad):
        '''
        @brief adjust phase of i and q baseband signals
        @param[in] qam_signal - signal to adjust
        @param[in] phase_rot_rad- rotation in radians
        '''
        adjusted_i = np.zeros_like(qam_signal.i_baseband)
        adjusted_q = np.zeros_like(qam_signal.q_baseband)
        baseband = qam_signal.i_baseband+1j*qam_signal.q_baseband
        for i,val in enumerate(baseband):
           mag,phase = cmath.polar(val)
           phase += phase_rot_rad
           adj_iq = cmath.rect(mag,phase)
           adjusted_i[i] = adj_iq.real
           adjusted_q[i] = adj_iq.imag
        qam_signal.i_baseband = adjusted_i
        qam_signal.q_baseband = adjusted_q
        
    def _adjust_decoded_phase(self,qam_signal,phase_rot_rad):
        '''
        @brief adjust phase of decoded signals
        @param[in] qam_signal - signal to adjust
        @param[in] phase_rot_rad- rotation in radians
        '''
        adjusted_iq = np.zeros_like(qam_signal.decoded_iq,dtype=np.complex128)
        for i,val in enumerate(qam_signal.decoded_iq):
            mag,phase = cmath.polar(val)
            phase += phase_rot_rad
            adjusted_iq[i] = cmath.rect(mag,phase)
        qam_signal.decoded_iq = adjusted_iq
    
    
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
        self.clock_sine = None #pulse train for the sampling locations
        
        #decoded values
        self.decoded_iq = None
        self._decoded_i_baseband = None
        self._decoded_q_baseband = None
        self.decoded_bitstream = None
        self.phase_correction = None
        self.mag_correction = None
        
        #RF (upconverted) values
        self.rf_signal = None
        self.if_i = None #downconverted values from rf signal
        self.if_q = None 
        
    @property
    def clock(self):
        '''
        @brief get a clock pulse train from sinusoid zero crossings
        '''
        clk_tf = self.clock_sine>=0
        clk_diff = np.diff(clk_tf)
        clk_diff = np.append(clk_diff,False)
        clk = clk_diff.astype(np.int8)
        return clk
        
    def plot_rf(self):
        '''
        @brief plot the upconverted rf signal
        '''
        plt.figure()
        plt.plot(self.times,self.rf_signal)
        return plt.gca()
    
    def plot_baseband(self):
        '''
        @brief plot the i and q baseband signals
        '''
        fig = plt.figure()
        plt.plot(self.times,self.i_baseband,label="I Baseband")
        plt.plot(self.times,self.q_baseband,label="Q Baseband")
        plt_min = np.min([self.i_baseband.min(),self.q_baseband.min()])
        plt_max = np.max([self.i_baseband.max(),self.q_baseband.max()])
        plt.plot(self.times,self.clock*(plt_max-plt_min)+plt_min)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')
        return fig
    
    def plot_decoded_baseband(self):
        '''
        @brief plot the decoded i and q baseband signals
        '''
        fig = plt.figure()
        plt.plot(self.times,self._decoded_i_baseband,label="I Baseband")
        plt.plot(self.times,self._decoded_q_baseband,label="Q Baseband")
        plt_min = np.min([self._decoded_i_baseband.min(),self._decoded_q_baseband.min()])
        plt_max = np.max([self._decoded_i_baseband.max(),self._decoded_q_baseband.max()])
        plt.plot(self.times,self.clock*(plt_max-plt_min)+plt_min)
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
    
    def plot_decoded_iq(self,ax=None):
        '''
        @brief plot the current _decoded_i_baseband and _decoded_q_baseband onto a 2d iq plot
        @param[in/OPT] ax - axis to plot on. if none make one
        '''
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(self._decoded_i_baseband,self._decoded_q_baseband)
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
    myqm = None
    myqam = None
    myqm = QAMModem(16)
    fig=myqm.plot_constellation()
    #mymap,inbits = myq.map_to_constellation(bytearray('testing'.encode()))
    #data = 'testing'.encode()
    data = np.random.random(5)
    #data = [255,255,255]
    myqam = myqm.encode_baseband(bytearray(data))
    #myqam.plot_baseband()
    #plt.plot(myqam.times,myqam.clock*2-1)
    #plt.plot(myqam.times,myqam.clock*2-1)
    myqm.decode_baseband(myqam)
    myqm.correct_iq(myqam)
    ax = fig.gca()
    #myqam.plot_decoded_iq(ax)
    #myqam.plot_decoded_points(ax)
    #myqam.plot_decoded_baseband()
    myqm.upconvert(myqam)
    myqm.downconvert(myqam)
    myqam.plot_rf()
    myqam.plot_downconverted_if()
    myevm = myqm.calculate_evm(myqam)
    print(myevm)
    #inbits = np.unpackbits(bytearray('testing'.encode()))
    #plt.plot(mymap.real,mymap.imag-0.01)
    #vals,outbits = myq.unmap_from_constellation(mymap)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
   
   
   
    