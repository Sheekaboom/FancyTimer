# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:45:52 2019

@author: aweis
"""

import numpy as np
#from pycom.modulation.Modulation import generate_gray_code_mapping
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cmath
import unittest
import itertools

#%% generic function for to and from complex to mag/phase
def complex2magphase(data):
    '''@brief Convert complex data to magnitude and phase'''
    return np.abs(data),np.angle(data)
    
def magphase2complex(mag,phase):
    '''@brief Convert magnitude and phase data to complex'''
    real = mag*np.cos(phase)
    imag = mag*np.sin(phase)
    return real+1j*imag

#%% map and unmap data to and from bitstream
def data2bitstream(data,**kwargs):
    '''
    @brief take an np.ndarray of data and change it to a bitstream. using np.unpackbits
    @param[in] data - np.ndarray of data to unpack into a bitstream.
    @param[in] kwargs - keyword args passed to np.unpackbits
    @todo remove intermediate bytearray step. Data is copied when that step is performed
    @return np.ndarray of bitstream from np.unpackbits
    '''
    mybytearray = bytearray(data)
    bitstream = np.unpackbits(mybytearray,**kwargs)
    return bitstream

def bitstream2data(bitstream,dtype,**kwargs):
    '''
    @brief take an np.ndarray of data and change it to a bitstream. using np.unpackbits
    @param[in] bitstream - np.ndarray bitstream (from unpackbits) to change to a dtype
    @param[in] dtype - dtype to create from the bits. if 'bitstream' or None just return the input
    @param[in] kwargs - keyword args passed to np.frombuffer
    @note this will round to the required number of full bits to remove any padding
    @todo remove intermediate bytearray step. Data is copied when that step is performed
    @return An np.ndarray of type specified by argument dtype
    '''
    #find the length of the dtype compared to uint8
    dtype_bytes = len(np.ndarray((1,),dtype=dtype).tobytes())
    if len(bitstream)%(dtype_bytes*8): #if they arent divisible
        bitstream = bitstream[:-(len(bitstream)%(dtype_bytes*8))] #remove some padding
    if dtype=='bitstream' or dtype is None:
        return bitstream 
    else:
        mybytearray = bytearray(np.packbits(bitstream))
        data = np.frombuffer(mybytearray,dtype=dtype,**kwargs)
        return data

#%% and now for our constellation generation functions
# This is done per the 3GPP spec for 5G NR (38.211 page 13)
def pad_bitstream(bitstream,bits_per_symbol):
    '''
    @brief pad a bitstream to fit the desired bits_per_symbol
    @param[in] bitstream - bitstream to pad
    @param[in] bits_per_symbol - number of bits per symbol in the constellation
    '''
    mod_val = len(bitstream)%bits_per_symbol
    if mod_val:
        pad_array = np.zeros(bits_per_symbol-(mod_val),dtype=np.uint8)
    else:
        pad_array = np.empty(0)
    return np.concatenate((bitstream,pad_array))

def get_permutation_bitstream(num_bits):
    '''
    @brief create a bitstream that covers every possible permutation of bits
        e.g (2 is 00,01,10,11)
    @param[in] num_bits - number of bits to create full coverage for
    @cite https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long
    ''' 
    bits = np.array(list(itertools.product([0,1],repeat=num_bits)),dtype=np.uint8) 
    return bits.flatten()  

def map_bpsk():
    '''
    @brief map function for bpsk. Returns a dict with binary/complex pairs
    @cite 3GPP rel 38.211 sec 5.1.2
    @param[in] bitstream - bitstream to map
    @return Dict mapping binary/complex iq locations
    '''
    bitstream = np.array([0,1])
    bi = bitstream  
    di = 1/np.sqrt(2)*((1-2*bi)+1j*(1-2*bi))  
    di_dict = {''.join(bits.astype(str)):vals for bits,vals in zip(bitstream.reshape(-1,1),di)}
    return di_dict

def map_qpsk():
    '''
    @brief map function for bpsk. Returns a dict with binary/complex pairs
    @cite 3GPP rel 38.211 sec 5.1.3
    @param[in] bitstream - bitstream to map
    @return Dict mapping binary/complex iq locations
    '''
    bps = bits_per_symbol = 2
    bitstream = get_permutation_bitstream(bps)
    bi = [bitstream[i::bps].astype(np.int32) for i in range(bps)]
    di = 1/np.sqrt(2)*((1-2*bi[0])+1j*(1-2*bi[1]))
    di_dict = {''.join(bits.astype(str)):vals for bits,vals in zip(bitstream.reshape(-1,bps),di)}
    return di_dict

def map_16qam():
    '''
    @brief map function for bpsk. Returns a dict with binary/complex pairs
    @cite 3GPP rel 38.211 sec 5.1.3
    @param[in] bitstream - bitstream to map
    @return Dict mapping binary/complex iq locations
    '''
    bps = bits_per_symbol = 4
    bitstream = get_permutation_bitstream(bps)
    bi = [bitstream[i::bps].astype(np.int32) for i in range(bps)]
    di = 1/np.sqrt(10)*((1-2*bi[0])*(2-(1-2*bi[2]))+
        1j*(1-2*bi[1])*(2-(1-2*bi[3])))
    di_dict = {''.join(bits.astype(str)):vals for bits,vals in zip(bitstream.reshape(-1,bps),di)}
    return di_dict


#%% and now the class to contain all of his
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
        M = self._set_map_fun(M) #parse M and set self._map_fun 
        self._constellation_dict = {} #dictionary containing key/values for binary/RI constellation
        self._generate_qam_constellation() #generate the constellation for M-QAM  

    def _set_map_fun(self,M):
        '''
        @brief set the mapping function for creating the qam constellation
        @param[in] M - modulation type to use. could be qam type (e.g., 16) or
            a string for the modulation (e.g.,bpsk,16qam)
        '''
        if isinstance(M,int): #then change to str
            M = str(M)+'qam'
        M = M.lower()
        map_dict = {
            '2qam'  :map_bpsk,
            'bpsk'  :map_bpsk,
            '4qam'  :map_qpsk,
            'qpsk'  :map_qpsk,
            '16qam' :map_16qam,
            }
        self._map_fun = map_dict.get(M,None)
        
    def _generate_qam_constellation(self):
        '''
        @brief generate an M-QAM constellation
        @param[in] M - order of QAM (must be power of 2)
        @note this checks self._map_fun function to create mapping from. 
            should return a dict with binary/complex key/value pairs
        @note if no function provided, pass for now
        @todo allow generic function for arbitrary M
        '''
        if self._map_fun is None:
            raise Exception('Mapping Function in not defined. Check input.')
            #self._constellation_dict = generate_gray_code_mapping(self.M,generate_qam_position)
            #avg_mag = self._get_average_constellation_magnitude()
            #self._constellation_dict.update((k,v/avg_mag) for k,v in self._constellation_dict.items())
        else:
            self._constellation_dict = self._map_fun() #this should return a dict
        #now normalize the average magnitude of the constellation to be 1
               
        
    def map(self,data,**arg_options):
        '''
        @brief map input data to constellation. If the number of bits does
        not fit in the encoding then pad with the value given by padding
        @param[in] data - data to map. currently supports 'uint'
            for raw bitstream use named argument 'bitstream' to True
        @param[in/OPT] arg_options - optional keyword arguments as follows:
            - None yet
        @return a list of complex numbers for the corresponding mapping, mapped bitstream
        '''
        options = {}
        for key,val in arg_options.items():
            options[key] = val
        bitstream = data2bitstream(data)
        if len(bitstream)==0: #if theres no data return an empty array
            return np.array([],dtype=np.csingle) #if we have no data for some reason (like all pilot tones) return empty array
        #if we have data finish mapping
        bits_per_symbol = self.bits_per_symbol
        bitstream = pad_bitstream(bitstream,bits_per_symbol)
        symbols = np.split(bitstream,len(bitstream)/bits_per_symbol) #split into symbols
        locations = [self._get_location(sym) for sym in symbols]
        return np.array(locations)
    
    def unmap(self,iq_data,dtype,correct_iq=None,**kwargs):
        '''
        @brief unmap a set of iq (complex) values from the constellation
        @param[in] iq_data - iq locations to unmap to bits
        @param[in] dtype - if specified return a numpy array of a set type. 
            Otherwise return a bytearray
        @param[in/OPT] correct_iq - correct constellation points if not provided simply assume the closest one
        @return a bytearray of the unmapped values
        '''
        if len(iq_data)==0:
            return np.array([],dtype=dtype),np.nan
        vals,err = self._get_values(iq_data,correct_locations=correct_iq)
        bitstream = vals.reshape((-1,)).astype('int')
        packed_vals = bitstream2data(bitstream,dtype)
        return packed_vals,err
    
    def _get_location(self,bits):
        '''
        @brief get a location (complex number) on the constellation given
            a set of bits. These bits should be an array (or list) of ones and
            zeros in LSB first order (how np.unpackbits provides values)
        @param[in] bits - list (or array) of bits in LSB first order (usually form np.unpackbits)
        '''
        #some checks
        bps = self.bits_per_symbol
        if len(bits) != bps:
            raise ValueError("Number of bits must equal Log2(M)")
        if not np.all(np.logical_or(bits==1,bits==0)):
            raise ValueError("Bits argument must be a list (or array) of only zeros or ones")
        key = np.array(np.flip(bits,axis=0),dtype=np.uint8) #MSB first for key
        key = ''.join(key.astype(str))
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
            values.append(np.flip(np.array(list(location_dict[matched_locations[i]])).astype(np.uint8),axis=0))
            #val = np.flip(location_dict[matched_locations[i]],axis=0)
            #values.append(val)
            #errors.append(cur_err)
        evm = self.calculate_evm(input_locations,matched_locations)
        return np.array(values),evm
    
    def calculate_evm(self,measured_values,correct_values):
        '''
        @brief calculate the evm from a signal. 
        @cite This is calulcated from equation (1)
            from "Analysis on the Denition Consistency Problem of EVM Measurement 
            and Its Solution" by Z. Feng, J. Riu, S. Jing-Lu, and Z. Xin
        @param[in] input_signal - QAMSignal with the ideal IQ locations in data_iq
        @param[in] output_signal - QAMSignal with the decoded IQ locations in data_iq
        '''
        evm_num = np.sum(np.abs(correct_values-measured_values)**2)
        evm_den = np.sum(np.abs(correct_values)**2)
        evm = np.sqrt(evm_num/evm_den)*100
        return evm
    
    def _get_average_constellation_magnitude(self):
        '''
        @brief return the average magnitude of points in the constellation
        @note this pulls the values from self.constellation_dict
        '''
        constellation_mag = [np.abs(v) for v in self._constellation_dict.values()]
        return np.mean(constellation_mag)
    
    @property
    def M(self):
        '''@brief Getter for backward comaptability'''
        return 2**self.bits_per_symbol
    
    @property
    def bits_per_symbol(self):
        '''@brief Getter for the number of bits per symbol'''
        return len(list(self._constellation_dict.keys())[0])
    
    @property
    def codes_dict(self):
        '''@brief getter for the coding of the qam'''
        codes = []
        for k,v in self.constellation_dict.items():
            codes.append(k)
        return np.array(codes)
    
    @property
    def locations_dict(self):
        '''@brief getter for the constellation locations (complex numbers)'''
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

    def plot_mpl(self,**arg_options):
        '''@brief plot the constellation points defined in constellation_dict with matplotlib'''
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
    
    def plot(self,**arg_options):
        fig = go.Figure()
        val_names = list(self._constellation_dict.keys())
        vals = np.array(list(self._constellation_dict.values()))
        fig.add_trace(go.Scatter(x=vals.real,y=vals.imag,text=val_names,
                                 mode='markers+text',textposition="bottom center",
                                 marker=dict(color=1, size=20),name='QAM Points'))
        return fig
    
#%% Class for adding false noise to iq data
class QAMError:
    '''
    @brief This is a class to add a variety of errors to QAM signals.
        Most methods here will be static methods that will simply take in
        and output an array of iq data points. Errors of various types will be added
        to these iq points
    '''
    
    @staticmethod
    def add_magnitude_noise(data,noise_funct):
        '''
        @brief Add magnitude errors to IQ data (linear)
        @param[in] data - np.ndarray of data to add errors to
        @param[in] noise_funct - function to generate the noise for the data.
            This function should take in a shape parameter (int or tuple of ints).
            An example is 'lambda shape: np.random.normal(0,1,shape)'
        @note All noise values will be added to linear magnitude values
        @return Data parameter with noise from noise_funct added on.
        '''
        mag,phase = complex2magphase(data)
        mag += noise_funct(data.shape)
        data_out = magphase2complex(mag, phase)
        return data_out
    
    @staticmethod
    def add_phase_noise(data,noise_funct):
        '''
        @brief Add phase errors to IQ data (radians)
        @param[in] data - np.ndarray of data to add errors to
        @param[in] noise_funct - function to generate the noise for the data.
            This function should take in a shape parameter (int or tuple of ints).
            An example is 'lambda shape: np.random.normal(0,1,shape)'
        @note All noise values will be added in radians
        @return Data parameter with noise from noise_funct added on.
        '''
        mag,phase = complex2magphase(data)
        phase += noise_funct(data.shape)
        data_out = magphase2complex(mag, phase)
        return data_out
    
    @staticmethod
    def add_i_noise(data,noise_funct):
        '''
        @brief Add noise to in phase (real part) of iq signal 
        @param[in] data - np.ndarray of data to add errors to
        @param[in] noise_funct - function to generate the noise for the data.
            This function should take in a shape parameter (int or tuple of ints).
            An example is 'lambda shape: np.random.normal(0,1,shape)'
        @return Data parameter with noise from noise_funct added on.
        '''
        data_out = data + noise_funct(data.shape)
        return data_out
        
    @staticmethod
    def add_q_noise(data,noise_funct):
        '''
        @brief Add noise to quadrature (imag part) of iq signal 
        @param[in] data - np.ndarray of data to add errors to
        @param[in] noise_funct - function to generate the noise for the data.
            This function should take in a shape parameter (int or tuple of ints).
            An example is 'lambda shape: np.random.normal(0,1,shape)'
        @return Data parameter with noise from noise_funct added on.
        '''
        data_out = data + 1j*noise_funct(data.shape)
        return data_out
    
    
#%% Other functions to use
def generate_qam_position(code_number,num_codes,normalize=True):
    '''
    @brief generate a qam position based on a given code number and total number of codes
    This is typically passed to generate_gray_code_mapping()
    This currently only works for a power of 2
    '''
    row_len = round(np.sqrt(num_codes)) #get the length of the side (e.g. 4 for 16 QAM)
    col_len = row_len
    #normalize size from -1 to 1
    if normalize:
        step_size = 1/(((row_len-1)/2))
    else:
        step_size = 1
    imag_part = (code_number%row_len)*step_size-1
    real_part = (np.floor(code_number/row_len)*step_size)-1
    return complex(real_part,imag_part)

class TestQamConstellation(unittest.TestCase):
    '''@brief Unit tests to test our Qam mapping functions'''

    def test_bitstream_conversion(self):
        '''@brief Test conversion of data to and from bitstreams'''
        data_len = 100;
        dtypes = [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32
                    ,np.float32,np.float64,np.complex64,np.complex128]
        data_list = [np.random.rand(data_len).astype(dt) for dt in dtypes]
        for data_in,dtype in zip(data_list,dtypes):
            bitstream = data2bitstream(data_in)
            data_out  = bitstream2data(bitstream,dtype)
            self.assertTrue(np.all(data_in==data_out))
            
    def test_map_unmap(self):
        '''@brief test mapping/unmapping to constellations'''
        m_vals = ['bpsk','qpsk','16qam']
        bits_per_symbol = [1,2,4]
        data_len = 1000
        mydtype = np.float16
        bytes_per_data = len(np.zeros((1),dtype=mydtype).tobytes())
        data_in = np.random.rand(data_len).astype(mydtype)
        for m,bps in zip(m_vals,bits_per_symbol):
            const = QAMConstellation(m)
            iq_data  = const.map(data_in)
            self.assertEqual(len(iq_data), (bytes_per_data*data_len*8)/bps)#ensure correct length
            data_out,err = const.unmap(iq_data, mydtype)
            self.assertTrue(np.all(data_in==data_out),msg='Failed on {}'.format(m))
            
            
            
    
if __name__=='__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQamConstellation)
    unittest.TextTestRunner(verbosity=2).run(suite)
     
    #for m in m_vals:
        #const = QAMConstellation(m)
        #const.plot()
    #b = map_bpsk()
    #q = map_qpsk()
    #s = map_16qam()
    #bs = get_permutation_bitstream(3)
    myqam = QAMConstellation(16)
    fig = myqam.plot()
    
    
    

    
    

    
   
   
   
   
   
    