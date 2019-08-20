# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from Modulation import Modulation
from Modulation import generate_gray_code_mapping
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

class QAM(Modulation):
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
    def __init__(self,M=None,**arg_options):
        '''
        @brief initialize the modulation class
        @param[in/OPT] M - type of QAM (e.g 4,16,256)
        '''
        defaults = {} #default options
        arg_options.update(defaults)
        super().__init__(**arg_options)
        self.M = M #what qam we are using
        self.constellation_dict = {} #dictionary containing key/values for binary/RI constellation
        if M is not None:
            self.generate_mqam_constellation(M)
        
    def generate_mqam_constellation(self,M):
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
    myq = QAM(64)
    myq.plot_constellation()
    mymap,inbits = myq.map_to_constellation(bytearray('testing'.encode()))
    #inbits = np.unpackbits(bytearray('testing'.encode()))
    plt.plot(mymap.real,mymap.imag-0.01)
    vals,outbits = myq.unmap_from_constellation(mymap)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
   
   
   
    