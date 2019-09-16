# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from Modulation import Modem
from Modulation import generate_gray_code_mapping,generate_root_raised_cosine,lowpass_filter
import numpy as np
import matplotlib.pyplot as plt
import cmath

class QAM
    
class QAMConstellation():
    '''
    @brief class to map, unmap, and hold a qam constellation
    '''
    def __init__(self,M,**arg_options):
        '''
        @brief initialize the modulation class
        @param[in/OPT] M - type of QAM (e.g 4,16,256)
        '''
        self.M = M
        self.constellation_dict = {} #dictionary containing key/values for binary/RI constellation
        self._generate_qam_constellation() #generate the constellation for M-QAM
        
    def map(self,data,**arg_options):
        '''
        @brief map input data to constellation. If the number of bits does
        not fit in the encoding then pad with the value given by padding
        @param[in] data - data to map. currently supports 'uint'
            for raw bitstream use named argument 'bitstream' to True
        @param[in/OPT] arg_options - optional keyword arguments as follows:
            padding - value to pad bits with if encoding doesnt fit (default 0)
;            bitstream - True if the data is a bitstream (array of 1s and 0s) (default False)
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
            loc = self._get_location(pack)
            locations.append(loc)
        return np.array(locations)
    
    def unmap(self,locations):
        '''
        @brief unmap a set of iq (complex) values from the constellation
        @param[in] locations - iq locations to unmap to bits
        @return a bytearray of the unmapped values, array of bits (bitstream)
        '''
        vals = self._get_values(locations)
        bitstream = vals.reshape((-1,)).astype('int')
        packed_vals = bytearray(np.packbits(bitstream))
        return packed_vals,bitstream
    
    def _get_location(self,bits):
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
    
    def _get_values(self,locations):
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
    
    def _generate_qam_constellation(self):
        '''
        @brief generate an M-QAM constellation
        @param[in] M - order of QAM (must be power of 2)
        '''
        self.constellation_dict = generate_gray_code_mapping(self.M,generate_qam_position)
    
    
    def plot(self,**arg_options):
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

    
    
    
    
    

    
    

    
   
   
   
   
   
    