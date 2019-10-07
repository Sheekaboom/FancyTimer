# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from pycom.modulation.Modulation import Modem
from pycom.modulation.Modulation import generate_gray_code_mapping
from pycom.modulation.Modulation import generate_root_raised_cosine,lowpass_filter
from samurai.analysis.support.MUFResult import complex2magphase,magphase2complex
import numpy as np
import matplotlib.pyplot as plt
import cmath
    
class QAMConstellation():
    '''
    @brief class to map, unmap, and hold a qam constellation
    '''
    def __init__(self,M,**arg_options):
        '''
        @brief initialize the modulation class
        @param[in] M - type of QAM (e.g 4,16,256)
        @param[in/OPT] arg_options - keyword arguments as follows:
            -None Yet!
        '''
        self.M = M
        self._constellation_dict = {} #dictionary containing key/values for binary/RI constellation
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
    
    def unmap(self,locations,dtype=None,correct_locations=None,**kwargs):
        '''
        @brief unmap a set of iq (complex) values from the constellation
        @param[in] locations - iq locations to unmap to bits
        @param[in/OPT] dtype - if specified return a numpy array of a set type. 
            Otherwise return a bytearray
        @param[in/OPT] correct_locations - correct constellation points if not provided simply assume the closest one
        @return a bytearray of the unmapped values
        '''
        vals,err = self._get_values(locations,correct_locations=correct_locations)
        bitstream = vals.reshape((-1,)).astype('int')
        packed_vals = bytearray(np.packbits(bitstream))
        if dtype is not None:
            packed_vals = np.frombuffer(packed_vals,dtype=dtype)
        return packed_vals,err
    
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
        val = self._constellation_dict[key]
        return val
    
    def _get_values(self,input_locations,correct_locations=None):
        '''
        @brief get a value (code) from a given constellation location(s) (complex number)
        @param[in] locations - complex number(s) for constellation value
        @param[in/OPT] correct_locations - correct constellation points if not provided simply assume the closest one
        @return np(array) of bit values, numpy array of complex errors from closest key
        '''
        if not hasattr(input_locations,'__iter__'):
            input_locations = [input_locations]
        location_dict = self._location_constellation_dict
        location_arr = np.array(list(location_dict.keys())) #list of locations
        matched_locations = np.zeros_like(input_locations) #the correct locations (if not provided)
        values = []
        for i,l in enumerate(input_locations):
            if correct_locations is not None: #then we use the provided correct location
                matched_locations[i] = correct_locations[i] #the correct location
            else:
                diffs = location_arr-l #complex distance from each location
                min_arg = np.argmin(np.abs(diffs))
                matched_locations[i] = location_arr[min_arg] #get the key to the closest constellation point
            values.append(np.flip(location_dict[matched_locations[i]],axis=0))
            #val = np.flip(location_dict[matched_locations[i]],axis=0)
            #values.append(val)
            #errors.append(cur_err)
        evm = self.calculate_evm(input_locations,matched_locations)
        return np.array(values),evm
    
    def calculate_evm(self,measured_values,correct_values):
        '''
        @brief calculate the evm from a signal. This is calulcated from equation (1)
            from "Analysis on the Denition Consistency Problem of EVM Measurement 
            and Its Solution" by Z. Feng, J. Riu, S. Jing-Lu, and Z. Xin
        @param[in] input_signal - QAMSignal with the ideal IQ locations in data_iq
        @param[in] output_signal - QAMSignal with the decoded IQ locations in data_iq
        '''
        evm_num = np.abs((correct_values-measured_values)**2).sum()
        evm_den = np.abs(correct_values**2).sum()
        evm = np.sqrt(evm_num/evm_den)*100
        return evm
    
    def _generate_qam_constellation(self):
        '''
        @brief generate an M-QAM constellation
        @param[in] M - order of QAM (must be power of 2)
        @note we also normalize the average magnitude of the constellation to 1 here
        '''
        self._constellation_dict = generate_gray_code_mapping(self.M,generate_qam_position)
        #now normalize the average magnitude of the constellation to be 1
        avg_mag = self._get_average_constellation_magnitude()
        self._constellation_dict.update((k,v/avg_mag) for k,v in self._constellation_dict.items())
        
    def _get_average_constellation_magnitude(self):
        '''
        @brief return the average magnitude of points in the constellation
        @note this pulls the values from self.constellation_dict
        '''
        constellation_mag = [np.abs(v) for v in self._constellation_dict.values()]
        return np.mean(constellation_mag)
    
    
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
        for k,v in self._constellation_dict.items():
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
        for k,v in self._constellation_dict.items():
            locations.append(v)
        return np.array(locations)
    
    @property
    def _location_constellation_dict(self):
        '''
        @brief this returns a dictionary the same as constellation_dict but with
        the key/value pairs flipped
        '''
        location_map_dict = {}
        for k,v in self._constellation_dict.items():
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

'''
#these are redefined from MUFResult
def complex2magphase(data):
    return np.abs(data),np.angle(data)

def magphase2complex(mag,phase):
    real = mag*np.cos(phase)
    imag = mag*np.sin(phase)
    return real+1j*imag
'''

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

    
if __name__=='__main__':
    myqam = QAMConstellation(64)
    myqam.plot()
    
    
    

    
    

    
   
   
   
   
   
    