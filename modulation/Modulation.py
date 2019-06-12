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
        self.options['sample_frequncy']   = 100e9
        self.options['carrier_frequency'] = 1e9
        self.options['baud_rate']         = 1e6
   
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
   
   
   
   
   
   
   
