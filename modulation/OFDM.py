# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:40:46 2019

@author: ajw5
"""
import numpy as np
import copy
from scipy import interpolate 
import matplotlib.pyplot as plt
import warnings
import unittest

from pycom.modulation.Modulation import Modem,TestModem
from pycom.modulation.Modulation import ModulatedSignal, ModulatedSignalFrame
from pycom.modulation.Modulation import lowpass_filter_zero_phase,bandstop_filter
from pycom.modulation.Modulation import complex2magphase,magphase2complex

def get_correction_from_pilots(pilot_freqs,meas_pilots,correct_pilots,**kwargs):
    '''
    @brief Take in pilot tone data and create a function to correct received data  
    @param[in] pilot_freqs - frequency list of each of the pilot tones
    @param[in] meas_pilots - measured pilot tone values at each pilot_freqs
    @param[in] correct_pilots - correct pilot tone values at each pilot_freqs
    @param[in/OPT] kwargs -keyword args as follows  
        - interp_type - type of interpolation to use for scipy.interp1d   
    @note interpolation done with mag/phase NOT real/complex
    @note this also assumes the pilots in packet are being matched to those in self['pilot_dict']
    @return a function to correct the packets for the current channel
        This allows different ofdm frames to be corrected for the same channel.
    '''
    options = {}
    options['interp_type'] = 'linear'
    for k,v in kwargs.items():
        options[k] = v
    #now lets extract the pilot subcarriers and data
    pp_mag,pp_phase = complex2magphase(meas_pilots) #recieved data
    pr_mag,pr_phase = complex2magphase(correct_pilots) #correct data
    p_mag_norm = pr_mag/pp_mag #normalized correction at each pilot
    p_phase_norm = pr_phase-pp_phase #normalized phase correction
    mag_interp_fun = interpolate.interp1d(pilot_freqs,p_mag_norm,kind=options['interp_type'],fill_value='extrapolate')
    phase_interp_fun = interpolate.interp1d(pilot_freqs,p_phase_norm,kind=options['interp_type'],fill_value='extrapolate')
    def channel_correct_funct(freqs,data,**kwargs):
        '''
        @brief function to correct a list of OFDMSignals to correct
        @param[in] freqs - frequencies for each data to correct in the packet data
        @param[in] data - data packet or list of data packets (np.ndarrays)
        '''
        out_data = []
        if np.ndim(data)<=0:
            data = [data]
        for d in data:
            mag_correction = mag_interp_fun(freqs)
            phase_correction = phase_interp_fun(freqs)
            cur_mag,cur_phase = complex2magphase(d)
            cur_mag*=mag_correction
            cur_phase+=phase_correction
            new_d = magphase2complex(cur_mag,cur_phase)
            out_data.append(new_d)
        return out_data
    return channel_correct_funct

class OFDMModem(Modem):
    '''
    @brief class to generate OFDM signals in frequency and time domain
        Frequency domain only simulation will be carried out py bypassing the 
        time domain steps involved. In a sense this still involves the time domain,
        but does not include any problems that may be caused by time domain effects.
        This will be produced as a set of multiple frequency domain packets that can
        be multiplied by a freqeuncy domain channel response. different packets 
        represent the time axis. This data flow will look as follows:
            - data input to bitstream
            - Modulate (from super())
            - set pilot tones on subcarriers. (part of serial2parallel)
            - serial2parallel (map to subcarriers)
            - multiply by frequncy domain channel response
            - Extract pilot tones
            - correct from pilot tones
            - parallel2serial (unmap to modulated stream)
            - demodulate (from super())
            
        Time domain simulation will be performed with a few added steps. In this 
        case the channel impulse response (CIR) will need to be estimated from
        the fft if frequency domain channel data is provided. In this case the data flow
        will look as follows:
            - data input to bitstream
            - Modulate (from super())
            - set pilot tones
            - serial2parallel (map to subcarriers)
            - ifft (zero padding to get correct length)
            - parallel to serial
            - convolve with time domain CIR
            - serial to parallel (?)
            - fft
            - Extract pilot tones
            - correct from pilot tones
            - parallel2serial (unmap from subcarriers to modulated stream)
            - demodulate (from super())
        
    @param[in] M - size of constellation (e.g. 16-QAM, M-QAM)
    @param[in] subcarrier_total - total number of subcarrier channels
        self.subcarrier_count removes pilot tone count
    @param[in] subcarrier_spacing - spacing between subcarriers in Hz
    @param[in] arg_options - keyword arguments as follows:
        - cp_length - length of the circular prefix (in ???)
            parameters passed to Modem Class
    '''
    def __init__(self,M,subcarrier_total,subcarrier_spacing,**arg_options):
        '''
        @brief constructor for the OFDMModem class
        @param[in] M - size of constellation (e.g. 16-QAM, M-QAM)
        @param[in] subcarrier_total - total number of subcarrier channels
            self.subcarrier_count removes pilot tone count
        @param[in] subcarrier_spacing - spacing between subcarriers in Hz
        @param[in] arg_options - keyword arguments as follows:
            - cp_length - length of the circular prefix (in ???)
                parameters passed to Modem Class
        '''
        super().__init__(M,baud_rate=1/subcarrier_spacing) #baud rate must be 1/carrier spacing for OFDM
        self['cp_length'] = .2 #default value (fraction of symbol length
        for key,val in arg_options.items():
            self.options[key] = val
        
        self['subcarrier_spacing'] = subcarrier_spacing
        self['subcarrier_total'] = subcarrier_total
        self['pilot_funct'] = None

#%% frequency domain operations    
    #Note that self.modulate and self.demodulate are part of super

    def serial2parallel(self,data,**kwargs):
        '''
        @brief Take serial iq data and parallelize it across our subcarriers
        @note This will add pilot tones based on the function self['pilot_funct']. 
            This function should be of the form pilot_funct(empty_packet,packet_num)
            where empty_packet is a np.ndarray of length self.subcarrier_total and packet_num
            is which packet we are on (allowing time signal changes). This should then return
            empty_packet with the inserted pilot tones. Pilot tones cannot be the same as the
            variable 'not_set_val' in this function which is complex(123,321).
        @param[in] data - complex modulated iq data to parallelize.
        @param[in/OPT] kwargs - keyword args as follows
            - pilot_funct - function for the pilot tones to override self['pilot_funct'].
                data must be deserialized with this same function
        @return list of np.ndarrays for each packet that has been parallelized
        '''
        options = {} #parse kwargs
        options['pilot_funct'] = self['pilot_funct']
        for k,v in kwargs.items():
            options[k] = v
        #now actually parallelize
        packet_data_list = [] #list of data to go into packets
        not_set_val = complex(123,321) #value indicating a carrier has not been set. Pilot tones cannot be this value
        used_data = 0 #amount of data that has been used
        open_sc_runaway_count = 0 #count how many times open_sc is 0 to find runaway (bad pilot funct)
        packet_num = 0 #what packet we are on
        while used_data<len(data): #go until weve set all of our data
            packet_data = np.ones((self.subcarrier_total,),dtype=np.cdouble)*not_set_val
            packet_data = options['pilot_funct'](packet_data,packet_num) #add pilot tones
            open_sc = len(packet_data[packet_data==not_set_val]) #number of open subcarriers
            if not open_sc: #check for multiple opensc values in a row
                open_sc_runaway_count += 1
                if open_sc_runaway_count > 10: #10 0 open packets in a row
                    raise Exception("Error in parallelization. 10 packets in a row with no room for data. Check pilot_funct")
            if used_data+open_sc >= len(data): #if we dont have enough data
                padding_data = np.zeros((used_data+open_sc-len(data)),dtype=np.cdouble) #pad the packet data
                data_for_packet = np.concatenate((data[used_data:],padding_data)) #concatenate our data with padded data
            else: # just use our data
                data_for_packet = data[used_data:used_data+open_sc]
            packet_data[packet_data==not_set_val] = data_for_packet #copy data to our packet
            packet_data_list.append(packet_data) #add our packet data to a list
            used_data += open_sc
            packet_num += 1
        return packet_data_list

    def get_pilot_tones(self,packet_data_list,**kwargs):
        '''
        @brief Get the pilot tones for our channel correction from a list of data from packets
        @param[in] packet_data_list - list of np.ndarrays containing data from each packet 
            with pilot_tones still included
        @param[in/OPT] kwargs - keyword args as follows:
            - pilot_funct - function for the pilot tones to override self['pilot_funct']. This needs to be the
                same as the function for which tones were added.
        @note this does not delete the pilot tones from the signal, only gets the values.
            Deletion is performed in the parallel2serial method.
        @return list for each packet of: subcarrier pilot tone indices, recieved pilot tone values, expected pilot tone values
        '''
        options = {} #parse kwargs
        options['pilot_funct'] = self['pilot_funct']
        for k,v in kwargs.items():
            options[k] = v
        #now lets get our pilot tones
        pilot_tone_idx_list      = []
        pilot_tone_recieved_list = []
        pilot_tone_expected_list = []
        not_set_val = complex(123,321) #this is a value that the pilot tones cannot be equal to

        for packet_num,packet_data in enumerate(packet_data_list): #loop through each packet
            #get where our pilot_tones are for this packet (assume pilot_tones are != not_set_val)
            pilot_finder = np.ones_like(packet_data)*not_set_val #get a known array
            pilot_finder = options['pilot_funct'](pilot_finder,packet_num)
            pilot_tone_idx = np.where(pilot_finder!=complex(123,321)) #where pilot finder has not been changed is not a pilot tone
            pilot_tone_rx  = packet_data[pilot_tone_idx]    #get the received values
            pilot_tone_exp = pilot_finder[pilot_tone_idx]   #get the expected values
            #now append
            pilot_tone_idx_list.append(pilot_tone_idx[0])
            pilot_tone_recieved_list.append(pilot_tone_rx)
            pilot_tone_expected_list.append(pilot_tone_exp)

        return pilot_tone_idx_list, pilot_tone_recieved_list, pilot_tone_expected_list

    def parallel2serial(self,packet_data_list,**kwargs):
        '''
        @brief Change parallel data to serial on recieve
        @param[in] packet_data_list - list of np.ndarrays containing data from each packet 
            with pilot_tones still included
        @param[in/OPT] kwargs - keyword args as follows:
            - pilot_funct - function for the pilot tones to override self['pilot_funct']. This needs to be the
                same as the function for which tones were added.
        @note Pilot tones cannot be the same as the variable 'not_set_val' in this function which is complex(123,321).
        @return np.ndarray of complex modulated iq data
        '''
        options = {} #parse kwargs
        options['pilot_funct'] = self['pilot_funct']
        for k,v in kwargs.items():
            options[k] = v
        #now lets remove our pilot tones
        data_out = np.ndarray(0,dtype=np.cdouble)
        not_set_val = complex(123,321) #this is a value that the pilot tones cannot be equal to
        for packet_num,packet_data in enumerate(packet_data_list): #loop through each packet
            #get where our pilot_tones are for this packet (assume pilot_tones are != not_set_val)
            pilot_finder = np.ones_like(packet_data)*not_set_val #get a known array
            pilot_finder = options['pilot_funct'](pilot_finder,packet_num)
            #where pilot finder has not been changed is not a pilot tone
            data_out = np.concatenate((data_out,packet_data[pilot_finder==complex(123,321)])) 
        return data_out
        

#%% Old operations
    
    def generate_ofdm_signal(self,data):
        '''
        @brief generate OFDMSignal packetized version of the data
        '''
        iq_points = self.constellation.map(data) #map data bytearray to iq constellation
        packets = np.array(self.packetize_iq(iq_points))
        mysig = OFDMSignal(data)
        return mysig
        
    def packetize_iq(self,iq_points,**kwargs):
        '''
        @brief take a set of iq points (complex numbers) and split them into packets for each time period
            The length of each packet will be equal to the number of subcarriers
            Extra channels in last packet will be filled with 'padding' value
        @param[in] iq_points - list of iq points to split up
        @param[in] kwargs - keyword args as follows
            padding - complex number to pad unused channels with (when required)
        @return list of numpy arrays with each of the channel values, padded iq_points
        '''
        options = {}
        options['padding'] = complex(0,0)
        options['dtype'] = np.csingle
        for k,v in kwargs.items():
            options[k] = v
        num_pts = len(iq_points)
        iq_points = np.array(iq_points)
        #change the count based on pilot tones
        pack_list = OFDMSignalFrame() #list of our packets to send. If we just have pilots, only returns a single pilot packet
        if not self.subcarrier_count: #if we have all pilots, make the first packet all pilots and the rest all data
            local_subcarrier_count = self.subcarrier_total #if we have all pilots, all packets after the first will be all data (first will be all pilots)
        else:
            local_subcarrier_count = self.subcarrier_count
            
        mod_val = (len(iq_points)%local_subcarrier_count) #amount of padding required
        if mod_val!=0:
            num_pad = local_subcarrier_count-mod_val
            print("Warning {} iq points not divisible by {} channels. Padding end with {}".format(num_pts,local_subcarrier_count,options['padding']))
            pad = np.full((num_pad,),options['padding'])
            iq_points = np.concatenate((iq_points,pad))
        if len(iq_points)>0:
            packets = np.split(iq_points,iq_points.shape[0]/local_subcarrier_count)
        else:
            packets = []
        #add pilot tones
        packets = self._add_pilot_tones_to_packet_data(packets) #still just list of data at this point
        for i,p in enumerate(packets):
            sig_opt = {'pilot_dict':copy.deepcopy(self['pilot_dict'])}
            if self.subcarrier_count==0 and i!=0: sig_opt['pilot_dict']=None #if all pilots and not first packet 
            pack_list.append(OFDMSignal(self.subcarriers,p,**sig_opt))
        return pack_list,iq_points
    
    def depacketize_iq(self,pack_list,**kwargs):
        '''
        @brief take a set of packets and change to a set of iq points
        @param[in] pack_list - list of Modulated packets to depacketize
        '''
        iq_points = np.array([])
        if self.subcarrier_count==0: #if all pilot tones are specified
            #remove the first packet as it is a pilot packet
            pack_list = pack_list[1:]
        for p in pack_list:
            dat = p.raw
            dat = self._del_pilot_tones_from_packet_data([dat])[0] #remove pilot tones
            iq_points = np.concatenate((iq_points,dat))
        return iq_points
    
    def set_pilot_tones(self,subcarrier_idx,tone_function=None):
        '''
        @brief create pilot tones given a list of subcarrier indices and a function
        @note if this function is None default is used (constant values)
        @param[in] subcarrier_idx - list of subcarrier indices for where to add pilot_tones
        @param[in] tone_function - function to generate tone from based on subcarrier idx
        @note a tone of a single value can be returned with something like lambda
        @note this sets self['pilot_dict']
        @todo decide whether these tones should be data (works for all constellations)
            or complex numbers that may or may not be a valid point in the constellation
        '''
        if tone_function is None:
            tone_function = lambda idx:complex(1,1)
        if type(subcarrier_idx)==str:
            if subcarrier_idx=='all': #then make make everything a pilot tone
                subcarrier_idx = np.range(self['subcarrier_total'])
        pilot_dict = {str(idx):tone_function(idx) for idx in subcarrier_idx}
        self['pilot_dict'] = pilot_dict
        
    def _add_pilot_tones_to_packet_data(self,pack_list):
        '''
        @brief add pilot tones to a set of packet data
        @param[in] pack_list - list of packets. self.subcarriers removes number of pilot tones
            from the count
        @return new numpy array of packets with pilots from self.pilot_dict
        '''
        out_pack_list = []
        if self.subcarrier_count==0: #all pilots
            #if we have all pilot tones then just make the first packet all pilots then the rest data
            out_pack_list.append([self.pilot_dict[k] for k in sorted(self.pilot_dict.keys())]) #create packet of all pilot
            for p in pack_list:
                out_pack_list.append(p)
        else: #interleaved pilots
            for p in pack_list:
                for ki in sorted([int(k) for k in self.pilot_dict.keys()]): #add keys from lowest to highest
                    k = str(ki)
                    #doing it like this allows us to ensure that we have the final index
                    #although this is definitely not efficient
                    p = np.insert(p,ki,self.pilot_dict[k])
                out_pack_list.append(p)
        return np.array(out_pack_list)
    
    def _del_pilot_tones_from_packet_data(self,pack_list):
        '''
        @brief remove pilot tones from a set of packet data
        @param[in] pack_list - remove pilot tones from data NOT OFDMPacket just numpy array
        @note uses self.pilot_dict for indices of pilots
        '''
        pack_list_out = []
        idx_list = [int(k) for k in self.pilot_dict.keys()]
        if self.subcarrier_count==0: 
            #if all pilots are specified, we have a single pilot packet so
            # do not remove any pilot tones from the rest of the packets
            idx_list = []
        for p in pack_list:
            p = np.delete(p,idx_list)
            pack_list_out.append(p)
        return np.array(pack_list_out)
        
    def get_channel_correction_funct_from_pilots(self,packet,**kwargs):
        '''
        @brief take our input complex packet data and linearize the channel from it  
        @param[in] packet - packet of OFDMPacket type with matching pilots to self['pilot_dict']  
        @param[in/OPT] kwargs -keyword args as follows  
            - interp_type - type of interpolation to use for scipy.interp1d   
            - use_packet_pilots - use pilot mapping from packet.metadata['pilot_tones'] if available
        @note interpolation done with mag/phase NOT real/complex
        @note this also assumes the pilots in packet are being matched to those in self['pilot_dict']
        @return a function to correct the packets for the current channel
            This allows different ofdm frames to be corrected for the same channel.
        '''
        options = {}
        options['interp_type'] = 'linear'
        options['use_packet_pilots'] = True
        for k,v in kwargs.items():
            options[k] = v
        if hasattr(packet,'metadata'): #get the pilot tones from our packet
            pt = packet.metadata.get('pilot_tones',None)
            if pt is not None: pilot_tones = pt #set our pilots from our packet if provided
            else: pilot_tones = self['pilot_dict'] #otherwise set from modem default
        else: #if not available in packet set from modem defualt
            pilot_tones = self['pilot_dict']
        #now lets extract the pilot subcarriers and data
        pilot_subcarriers = [packet.subcarriers[idx] for idx in pilot_tones.keys()]
        pilot_packet_data = [packet.data[idx] for idx in pilot_tones.keys()] #extract pilot data
        pilot_real_data = [pd for pd in pilot_tones.values()] #extract the correct data
        pp_mag,pp_phase = complex2magphase(pilot_packet_data) #recieved data
        pr_mag,pr_phase = complex2magphase(pilot_real_data) #correct data
        p_mag_norm = pr_mag/pp_mag #normalized correction at each pilot
        p_phase_norm = pr_phase-pp_phase #normalized phase correction
        mag_interp_fun = interpolate.interp1d(pilot_subcarriers,p_mag_norm,kind=options['interp_type'],fill_value='extrapolate')
        mag_correction = mag_interp_fun(packet.subcarriers)
        phase_interp_fun = interpolate.interp1d(pilot_subcarriers,p_phase_norm,kind=options['interp_type'],fill_value='extrapolate')
        phase_correction = phase_interp_fun(packet.subcarriers)
        def channel_correct_funct(pack_list,**kwargs):
            '''
            @brief function to correct a list of OFDMSignals to correct
            @param[in] pack_list - list of OFDMSignals to correct
            '''
            if not isinstance(pack_list,list):
                pack_list = [pack_list]
            out_packs = []
            for p in pack_list:
                cur_mag,cur_phase = complex2magphase(p.data)
                cur_mag*=mag_correction
                cur_phase+=phase_correction
                cur_data = magphase2complex(cur_mag,cur_phase)
                new_p = copy.deepcopy(p)
                new_p.data = cur_data
                out_packs.append(new_p)
            return out_packs
        return channel_correct_funct,mag_correction,phase_correction
    
    @property
    def pilot_count(self):
        '''@brief getter for number of pilot tones'''
        return len(self['pilot_dict'])
    
    @property
    def subcarrier_count(self):
        '''@brief return self['subcarrier_total']-self.pilot_count'''
        return self['subcarrier_total']-self.pilot_count
    
    @property
    def subcarriers(self):
        '''@brief return a numpy array of our subcarriers'''
        return np.arange(self.subcarrier_total)*self.subcarrier_spacing
    
#%% time domain operations
        
    def modulate_data_time_domain(self,data):
        '''
        @brief take a set of data and generate an OFDM modulated signal from this
        @return a OFDMSignal object containing the data of the signal
        '''
        #split the data across our channels
        mysig = self._generate_ofdm_signal(data)
        mysig.packets = packets
        baseband_signal_td = np.fft.ifft(packets,axis=0)
        mysig.baseband_dict.update({'i':baseband_signal_td.real})
        mysig.baseband_dict.update({'q':baseband_signal_td.imag})
        dt = self.baud_rate/self.subcarrier_count
        mysig.baseband_times = np.arange(0,self.baud_rate,dt)
        #now oversample the signal to match that of our high frequency value
        self._oversample(mysig)
        return mysig
        
    def _oversample(self,ofdm_signal):
        '''
        @brief create an oversampled baseband signal at the rate of self.options['sample_frequency']
        @param[in] ofdm_signal - OFDMSignal object containing a undersampled baseband time domain signal
        '''
        #create our oversampled times and iq full of zeros
        ofdm_signal.times = np.arange(0,self.baud_rate,1/self.sample_frequency)
        i_oversampled = np.zeros((ofdm_signal.baseband_dict['i'].shape[0],ofdm_signal.times.shape[0]))
        q_oversampled = np.zeros((ofdm_signal.baseband_dict['q'].shape[0],ofdm_signal.times.shape[0]))
        #now generate a set of steps on our oversampled data
        for pnum,pack in enumerate(ofdm_signal.packets): #and do for each packet
            for num,t in enumerate(ofdm_signal.baseband_times):
                #go through each baseband time for oversampling
                #probably not the best way to do this. we are doing a lot of overwriting
                    cur_i = ofdm_signal.baseband_dict['i'][pnum][num]
                    cur_q = ofdm_signal.baseband_dict['q'][pnum][num]
                    i_oversampled[pnum][ofdm_signal.times>=t] = cur_i
                    q_oversampled[pnum][ofdm_signal.times>=t] = cur_q
            
        ofdm_signal.baseband_dict.update({'i_oversampled':i_oversampled})
        ofdm_signal.baseband_dict.update({'q_oversampled':q_oversampled})
    
    def upconvert(self,ofdm_signal):
        '''
        @brief unpocvert a signal to the set carrier frequency
        '''
        fc = self.options['carrier_frequency']
        II = ofdm_signal.baseband_dict['i_oversampled']* np.cos(2.*np.pi*fc*ofdm_signal.times);
        QQ = ofdm_signal.baseband_dict['q_oversampled']*-np.sin(2.*np.pi*fc*ofdm_signal.times);
        IQ = II+QQ;
        ofdm_signal.rf_signal = IQ
        
    def ideal_upconvert(self,ofdm_signal):
        '''
        @brief ideal unconversion in frequency domain of the ofdm signal
        @param[in] ofdm_packets - ofdm signal class with frequency domain packets
        '''
        raise NotImplementedError
        
    def downconvert(self,ofdm_signal):
        '''
        @brief downconvert a OFDM signal object with only an RF signal to i_oversampled and q_oversampled
        '''
        fc = self.options['carrier_frequency']
        i_bb =  ofdm_signal.rf_signal* np.cos(np.pi*2.*fc*ofdm_signal.times);#/np.sum(np.square(np.cos(np.pi*2.*self.fc*times)))
        q_bb =  ofdm_signal.rf_signal*-np.sin(np.pi*2.*fc*ofdm_signal.times);#/np.sum(np.square(np.sin(np.pi*2.*self.fc*times)))

        #lowpass filter with zero phase change
        #i_bb = lowpass_filter_zero_phase(i_bb,1/self.options['sample_frequency'],1e9)
        i_bb = bandstop_filter(i_bb,1/self.sample_frequency,2*self.carrier_frequency)
        
        ofdm_signal.baseband_dict['i_oversampled'] = i_bb
        ofdm_signal.baseband_dict['q_oversampled'] = q_bb
        #qam_signal.clock_sine = clk_sin
        
        baseband_dict = {'i':i_bb,'q':q_bb}
        return baseband_dict
        
    def _add_cp(self,ofdm_signal):
        '''
        @brief add the cyclic prefix to the signal
        @param[in] ofdm_signal - OFDMSignal object with populated time domain baseband signals
        cp length is given as a fraction of baud rate
        This is done before oversampling so we still have full fft bins
        @todo finish this
        '''
        #get the indices of our values to put in the time
        for pack_num in range(len(ofdm_signal.packets)):
            cp_time = self.cp_length*self.baud_rate
            cp_i_vals = ofdm_signal.baseband_dict['i'][pack]

class OFDMSignalFrame(ModulatedSignalFrame):
    '''
    @brief class to hold list of OFDM signals (frame)
    @note there is nothing new over ModulatedSignalFrame. Just a naming
    '''
    pass

class OFDMSignal(ModulatedSignal):
    '''
    @brief class to hold a generic ofdm modulated signal type
    @param[in] everything is passed to ModulatedSignal for init. all kwargs are saved to self.metadata
    @todo add pilots here (or pilot gen function), be able to generate channel correction fromm pilots saved here. Maybe also save the data being transmitted here
    '''
    
    def __init__(self,*args,**kwargs):
        '''@brief constructor'''
        super().__init__(*args,**kwargs)
    
    @property
    def subcarriers(self):
        '''@brief simply another name for freq_list. This packet should only have subcarriers though'''
        return self.freq_list
    
    
class OFDMError(Exception):
    pass

class OFDMPilotError(OFDMError):
    pass

class OFDMWarning(Warning):
    pass

class OFDMPilotWarning(OFDMWarning):
    pass


class TestOFDMModem(TestModem):
    '''@brief Unittest class for an OFDM Modem class'''

    def test_ser2par_par2ser(self):
        '''@brief Test serial2parallel and parallel2serial with an arbitrary pilot funct'''
        def pilot_funct(data,pack_num):
            '''@brief every 3rd tone is a pilot'''
            data[::3] = complex(1,1)
            return data
        mymodem = OFDMModem(16,10,60e3)
        mydtype = np.float16
        data_in  = np.random.rand(160).astype(mydtype)
        mod_data = mymodem.modulate(data_in)
        par_data = mymodem.serial2parallel(mod_data,pilot_funct=pilot_funct)
        ser_out  = mymodem.parallel2serial(par_data,pilot_funct=pilot_funct)
        data_out = mymodem.demodulate(ser_out,mydtype)
        self.assertTrue(np.all(data_in==data_out))
    
    def atest_packetize(self):
        '''
        @brief test OFDMModem.packetize_iq and OFDMModem.depacketize_iq symmetric property
        '''
        mymodem = OFDMModem(64,1666,60e3)
        data = np.random.rand(3332)#9996)
        mapped = mymodem.constellation.map(data)
        packs,mapped_in = mymodem.packetize_iq(mapped)
        mapped_out = mymodem.depacketize_iq(packs)
        self.assertTrue(np.all(mapped_in==mapped_out))
    
    def atest_encode(self):
        '''
        @brief full test for the following
            raw_data->QAM mapping->packetization->depacktization->qam demapping->raw_data
        @note this also tests adding in pilot tones
        '''
        subcarriers = 1666
        channel_spacing = 60e3
        qam_M = 64
        num_packs = 2#6
        pilot_step = int(subcarriers/8)
        pilot_idx = np.arange(pilot_step-1,subcarriers,pilot_step,dtype=np.int32)
        pilot_funct = lambda idx: complex(20+0j)
        #init our modem
        mymodem = OFDMModem(qam_M,subcarriers,channel_spacing)
        #set the pilot tones
        mymodem.set_pilot_tones(pilot_idx,pilot_funct)
        #set our data
        num_data = mymodem.subcarrier_count*6
        data = np.random.rand(num_data)
        data_in=data
        mapped = mymodem.constellation.map(data)
        packs,_ = mymodem.packetize_iq(mapped)
        from samurai.base.TouchstoneEditor import SnpParam
        channel = SnpParam(packs[0].freq_list,np.ones(packs[0].freq_list.shape))
        packs_out = [pack*channel for pack in packs]
        mapped_out = mymodem.depacketize_iq(packs_out)
        data_out,err = mymodem.constellation.unmap(mapped_out,dtype='float64')
        self.assertTrue(np.all(data_in==data_out))
        
    def atest_all_pilot(self):
        '''
        @brief test generating all pilot tones in a packet using mymodem.set_pilot_tones('all')
        '''
        subcarriers = 1666
        channel_spacing = 60e3
        qam_M = 64
        num_packs = 6
        pilot_step = 1
        pilot_idx = np.arange(pilot_step-1,subcarriers,pilot_step,dtype=np.int32)
        pilot_funct = lambda idx: complex(20+0j)
        #init our modem
        mymodem = OFDMModem(qam_M,subcarriers,channel_spacing)
        #set the pilot tones
        mymodem.set_pilot_tones(pilot_idx,pilot_funct)
        #set our data
        #num_data = mymodem.subcarrier_total*6
        num_data = 10
        data = np.random.rand(num_data)
        data_in=data
        mapped = mymodem.constellation.map(data)
        packs,mapped_in = mymodem.packetize_iq(mapped)
        from samurai.base.TouchstoneEditor import SnpParam
        channel = SnpParam(packs[0].freq_list,np.ones(packs[0].freq_list.shape))
        packs_out = [pack*channel for pack in packs]
        fun,_,_ = mymodem.get_channel_correction_funct_from_pilots(packs_out[0]) #correct from pilot
        mapped_out = mymodem.depacketize_iq(packs_out)
        data_out,err = mymodem.constellation.unmap(mapped_out,dtype='float64')
        self.assertTrue(np.all(mapped_in==mapped_out))
            
        
if __name__=='__main__':
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOFDMModem)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()

    import copy
    import os
    from Modulation import plot_frequency_domain
    import unittest
    
    def pilot_funct(data,pack_num):
        '''@brief every 4th tone is a pilot'''
        data[::4] = complex(1,1)
        return data
    mymodem = OFDMModem(16,10,60e3)
    mydtype = np.float16
    data = np.random.rand(160).astype(mydtype)
    mod_data = mymodem.modulate(data)
    par_data = mymodem.serial2parallel(mod_data,pilot_funct=pilot_funct)
    pilot_data = mymodem.get_pilot_tones(par_data,pilot_funct=pilot_funct)
    ser_out  = mymodem.parallel2serial(par_data,pilot_funct=pilot_funct)
    data_out = mymodem.demodulate(ser_out,mydtype)
    first_packet_pilots = tuple([d[0] for d in pilot_data])
    corr_fun = get_correction_from_pilots(*first_packet_pilots)
    
    from pycom.modulation.QAM import QAMError
    import plotly.graph_objs as go
    
    myerror = QAMError()
    qam_data = ser_out = ser_out[ser_out!=0]
    qam_data = myerror.add_phase_noise(qam_data,lambda shape: np.random.normal(0,.1,shape))
    qam_data = myerror.add_magnitude_noise(qam_data,lambda shape: np.random.normal(0,.1,shape))
    #qam_data = myerror.add_i_noise(qam_data,lambda shape: np.random.normal(0,.1,shape))
    #qam_data = myerror.add_q_noise(qam_data,lambda shape: np.random.normal(0,.1,shape))
    fig = mymodem.options['constellation'].plot()
    fig.add_trace(go.Scatter(x=qam_data.real, y=qam_data.imag,mode='markers'))
    

    '''
    subcarriers = 1666
    channel_spacing = 60e3
    qam_M = 16
    num_packs = 6
    pilot_step = 1
    pilot_idx = np.arange(pilot_step-1,subcarriers,pilot_step,dtype=np.int32)
    pilot_funct = lambda idx: complex(20+0j)
    #init our modem
    mymodem = OFDMModem(qam_M,subcarriers,channel_spacing)
    #set the pilot tones
    mymodem.set_pilot_tones(pilot_idx,pilot_funct)
    num_data = 10
    data = np.random.rand(num_data)
    data_in=data
    mapped = mymodem.constellation.map(data)
    packs,mapped_in = mymodem.packetize_iq(mapped)
    
    out_dir = r'Q:\public\Quimby\Students\Alec\PhD\ofdm_tests\ofdm_packets_64QAM_1666SC_27p5GHz'
    '''
    '''
    subcarriers = 1666
    channel_spacing = 60e3
    qam_M = 64
    num_packs = 6
    num_pilots = 833
    pilot_step = int(subcarriers/num_pilots)
    pilot_idx = np.arange(pilot_step,subcarriers,pilot_step,dtype=np.int32)
    pilot_funct = lambda idx: complex(3+0j)
    
    mymodem = OFDMModem(qam_M,subcarriers,channel_spacing)
    #mymodem = OFDMModem(256,7,60e3)
    data = 'testing'.encode()
    np.random.seed(1234) #reset seed for testing  
    
    #set our pilot tones
    mymodem.set_pilot_tones(pilot_idx,pilot_funct)
    
    num_data = mymodem.subcarrier_count*6*4
    data = np.random.rand(num_data).astype('float16')
    #data = np.random.rand(9996)
    mapped = mymodem.constellation.map(data)
    packs,mapped_in = mymodem.packetize_iq(mapped)
    
    p = packs[0]
    #corr_fun = mymodem.get_channel_correction_funct_from_pilots(p)
    #corr_pack = corr_fun([p])
    for i,p in enumerate(packs):
        p.S[21].freq_list+=27.674e9
        out_name = os.path.join(out_dir,'packet_{}.s1p'.format(i))
        print(out_name)
        p.write(out_name)
    #p.plot_iq(mymodem['constellation'])
    packs_out = []
    for p in packs:
        p.data*=complex(1,0)
        packs_out.append(p)
        
    Schannel = 1 #set to an SnpParam
    
    mapped_out = mymodem.depacketize_iq(packs_out)
    #data_out,err = mymodem.constellation.unmap(mapped_out,dtype='float64')
    data_out,err = mymodem.constellation.unmap(mapped_out,dtype=np.ubyte)
    #data = np.random.rand(10)
    #mysig = mymodem.modulate(data)
    #mymodem.upconvert(mysig)
    #mysig.plot_packet()
    #mysig.plot_baseband()
    
    #insig = mysig
    #outsig = copy.deepcopy(insig)
    
   # mymodem.downconvert(outsig)
    #outsig.plot_baseband()
    #plot_frequency_domain(outsig.baseband_dict['i_oversampled'][0],outsig.times[1]-outsig.times[0])
    #outsig.plot_baseband()
    '''
        
        
        
        
        
        
        
        