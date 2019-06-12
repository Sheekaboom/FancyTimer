# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
from Modulation import Modulation
import matplotlib.pyplot as plt

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
    def __init__(self,M,**arg_options):
        '''
        @brief initialize the modulation class
        @param[in] M - type of QAM (e.g 4,16,256)
        '''
        defaults = {} #default options
        arg_options.update(defaults)
        super().__init(**arg_options)
        self.iq_codes = [] #codes for matching iq positions
        self.iq_positions = np.empty([],dtype=np.complex64) #iq positions related to coding positions
       
    def plot_constellation(self,**arg_options):
        '''
        @brief plot the constellation of self.iq_positions and self.iq_codes
        '''
        
   
   #def modulate()
   #    pas 
   
def generate_gray_code_mapping(num_codes):
    '''
    @brief generate gray codes for a given number of codes
    '''
    num_bits = get_num_bits(num_codes-1)
    codes = []
    for i in range(num_codes): #for each desired code
        #generate our locations for iq here
        cur_code = np.zeros(num_bits)
        for bn in range(num_bits):
            cur_code[bn] = (((i+2**bn)>>(bn+1)))%2
        codes.append(cur_code)
    return np.array(codes)

def get_num_bits(number):
    '''
    @brief find the number of bits required for a number
    '''
    myba   = bytearray(np.array(number).byteswap().tobytes())
    mybits = np.unpackbits(myba)
    ones_loc = np.where(mybits==1)[0] #location of ones
    if ones_loc.shape[0] == 0: #if the list is empty
        return 0 #if the number is 0 we have 0 bits
    num_bits = mybits.shape[0] - ones_loc[0]
    return num_bits

if __name__=='__main__':
    
    codes = generate_gray_code_mapping(32)
    for c in np.flip(codes,axis=1):
        print(c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
   
   
   
    