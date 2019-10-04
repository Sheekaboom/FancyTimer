# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:40:46 2019

@author: ajw5
"""
from pycom.modulation.Modulation import Modem
from pycom.modulation.Modulation import ModulatedSignal,ModulatedPacket
from pycom.modulation.Modulation import lowpass_filter_zero_phase,bandstop_filter
from pycom.modulation.QAM import QAMConstellation
import numpy as np
import matplotlib.pyplot as plt

class OFDMModem(Modem):
    '''
    @brief class to generate OFDM signals in frequency and time domain
    '''
    def __init__(self,M,subcarrier_total,subcarrier_spacing,**arg_options):
        '''
        @brief constructor for the OFDMModem class
        @param[in] M - size of constellation (e.g. 16-QAM, M-QAM)
        @param[in] subcarrier_total - total number of subcarrier channels
            self.subcarrier_count removes pilot tone count
        @param[in] subcarrier_spacing - spacing between subcarriers in Hz
        @param[in] arg_options - keyword arguments as follows:
            cp_length - length of the circular prefix (in ???)
            parameters passed to Modem Class
        '''
        super().__init__(baud_rate=1/subcarrier_spacing) #baud rate must be 1/carrier spacing for OFDM
        self['cp_length'] = .2 #default value (fraction of symbol length
        for key,val in arg_options.items():
            self.options[key] = val
        
        self['constellation'] = QAMConstellation(M,**arg_options)
        self['subcarrier_spacing'] = subcarrier_spacing
        self['subcarrier_total'] = subcarrier_total
        self['pilot_dict'] = {}

        
    def modulate(self,data):
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
    
    def generate_ofdm_signal(self,data):
        '''
        @brief generate OFDMSignal packetized version of the data
        '''
        iq_points = self.constellation.map(data) #map data bytearray to iq constellation
        packets = np.array(self.packetize_iq(iq_points))
        mysig = OFDMSignal(data)
        return mysig
        
    def packetize_iq(self,iq_points,pilot_dict={},**kwargs):
        '''
        @brief take a set of iq points (complex numbers) and split them into packets for each time period
            The length of each packet will be equal to the number of subcarriers
            Extra channels in last packet will be filled with 'padding' value
        @param[in] iq_points - list of iq points to split up
        @param[in] pilot_dict - dictionary from self.create_pilot_tones to add to packets
        @param[in] kwargs - keyword args as follows
            padding - complex number to pad unused channels with (when required)
        @return list of numpy arrays with each of the channel values, padded iq_points
        '''
        options = {}
        options['padding'] = complex(0,0)
        for k,v in kwargs.items():
            options[k] = v
        num_pts = len(iq_points)
        iq_points = np.array(iq_points)
        #change the count based on pilot tones
        mod_val = (len(iq_points)%self.subcarrier_count) #amount of padding required
        if mod_val is not 0:
            num_pad = self.subcarrier_count-mod_val
            print("Warning {} iq points not divisible by {} channels. Padding end with {}".format(num_pts,self.subcarrier_count,options['padding']))
            pad = np.full((num_pad,),options['padding'])
            iq_points = np.concatenate((iq_points,pad))
        packets = np.split(iq_points,iq_points.shape[0]/self.subcarrier_count)
        pack_list = []
        subcarriers = np.arange(self.subcarrier_total)*self.subcarrier_spacing
        #add pilot tones
        packets = self._add_pilot_tones_to_packet_data(packets) #still just list of data at this point
        for p in packets:
            pack_list.append(OFDMPacket([p,subcarriers]))
        return pack_list,iq_points
    
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
        pilot_dict = {idx:tone_function(idx) for idx in subcarrier_idx}
        self['pilot_dict'] = pilot_dict
        
    def _add_pilot_tones_to_packet_data(self,pack_list):
        '''
        @brief add pilot tones to a set of packet data
        @param[in] pack_list - list of packets. self.subcarriers removes number of pilot tones
            from the count
        @return new numpy array of packets with pilots from self.pilot_dict
        '''
        out_pack_list = []
        for p in pack_list:
            for k in sorted(self.pilot_dict): #add keys from lowest to highest
                #doing it like this allows us to ensure that we have the final index
                #although this is definitely not efficient
                p = np.insert(p,k,self.pilot_dict[k])
            out_pack_list.append(p)
        return np.array(out_pack_list)
            
                
    @property
    def pilot_count(self):
        '''@breif getter for number of pilot tones'''
        return len(self['pilot_dict'])
    
    @property
    def subcarrier_count(self):
        '''@brief return self['subcarrier_total']-self.pilot_count'''
        return self['subcarrier_total']-self.pilot_count
    
    def depacketize_iq(self,pack_list,**kwargs):
        '''
        @brief take a set of packets and change to a set of iq points
        @param[in] pack_list - list of Modulated packets to depacketize
        '''
        iq_points = np.array([])
        for p in pack_list:
            dat = p.data
            dat = self._del_pilot_tones_from_packet_data([dat])[0] #remove pilot tones
            iq_points = np.concatenate((iq_points,dat))
        return iq_points
    
    def _del_pilot_tones_from_packet_data(self,pack_list):
        '''
        @brief remove pilot tones from a set of packet data
        @param[in] pack_list - remove pilot tones from data NOT OFDMPacket just numpy array
        @note uses self.pilot_dict for indices of pilots
        '''
        pack_list_out = []
        idx_list = list(self.pilot_dict.keys())
        for p in pack_list:
            p = np.delete(p,idx_list)
            pack_list_out.append(p)
        return np.array(pack_list_out)
        
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
        pass
        
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
        
        
class OFDMSignal(ModulatedSignal):
    '''
    @brief class to hold ofdm signal data
    '''
    def __init__(self,packet_list,**arg_options):
        '''
        @brief constructor for the class. Inherits from ModulatedSignal
        '''
        super().__init__(None,**arg_options)
        
        self['packets'] = packet_list
        for k,v in arg_options.items():
            self[k] = v
        
    def plot_packet(self,packet_number=0):
        '''
        @brief plot a packet in the frequency domain
        @param[in\OPT] packet_number - which packet to plot (default 0)
        '''
        fig = plt.figure()
        packet = self.packets[0]
        #I = packet.real
        #Q = packet.imag
        x = np.arange(len(packet))-(int(len(packet)/2))
        x = np.stack((x,x),axis=1)
        y = np.zeros(packet.shape)
        y = np.stack((y,y+np.abs(packet)),axis=1)
        plt.plot(x.transpose(),y.transpose(),label='Magnitude',color='b')
        return fig
    
    def plot_baseband(self,packet_number=0):
        '''
        @brief plot all baseband data. OFDM is weird so lets redo this
        '''
        fig = plt.figure()
        all_baseband = []
        for key,val in self.baseband_dict.items():
            if 'oversample' not in key:
                times = self.baseband_times
            else:
                times = self.times
            val = val[packet_number]
            plt.plot(times,val.transpose(),label='{} Baseband'.format(key))
            all_baseband.append(val)
            
        ax = fig.gca()
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')
        return fig
    
    def plot_rf(self,packet_number=0):
        '''
        @brief plot the upconverted rf signal
        '''
        plt.figure()
        plt.plot(self.times,self.rf_signal[packet_number])
        return plt.gca()

class OFDMPacket(ModulatedPacket):
    '''
    @brief class to hold a single OFDM packet. This will be frequency domain data
        which will represent a single OFDM frame
    '''
    def __init__(self,input_file=None,**kwargs):
        '''
        @brief constructor
        @param[in] input file - [[packet_data],[freq/time_list]] or string of path to touchstone file
        '''
        super().__init__(input_file,**kwargs)
    
    @property
    def subcarriers(self):
        '''
        @brief simply another name for freq_list. This packet should only have subcarriers though
        '''
        return self.freq_list
        
if __name__=='__main__':
    
    import copy
    import os
    from Modulation import plot_frequency_domain
    import unittest
    
    class OFDMTest(unittest.TestCase):
        
        def test_packetize(self):
            '''
            @brief test OFDMModem.packetize_iq and OFDMModem.depacketize_iq symmetric property
            '''
            mymodem = OFDMModem(64,1666,60e3)
            data = np.random.rand(9996)
            mapped = mymodem.constellation.map(data)
            packs,mapped_in = mymodem.packetize_iq(mapped)
            mapped_out = mymodem.depacketize_iq(packs)
            self.assertTrue(np.all(mapped_in==mapped_out))
        
        def test_encode(self):
            '''
            @brief full test for the following
                raw_data->QAM mapping->packetization->depacktization->qam demapping->raw_data
            '''
            subcarriers = 1666
            channel_spacing = 60e3
            qam_M = 64
            num_packs = 6
            pilot_step = int(subcarriers/8)
            pilot_idx = np.arange(pilot_step,subcarriers,pilot_step,dtype=np.int32)
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
            
    suite = unittest.TestLoader().loadTestsFromTestCase(OFDMTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()
    
    out_dir = r'Q:\public\Quimby\Students\Alec\PhD\ofdm_tests\ofdm_packets_64QAM_1666SC_27p5GHz'
    
    subcarriers = 1666
    channel_spacing = 60e3
    qam_M = 64
    num_packs = 6
    pilot_step = int(subcarriers/8)
    pilot_idx = np.arange(pilot_step,subcarriers,pilot_step,dtype=np.int32)
    pilot_funct = lambda idx: complex(20+0j)
    
    mymodem = OFDMModem(qam_M,subcarriers,channel_spacing)
    #mymodem = OFDMModem(256,7,60e3)
    data = 'testing'.encode()
    np.random.seed(1234) #reset seed for testing  
    
    #set our pilot tones
    mymodem.set_pilot_tones(pilot_idx,pilot_funct)
    
    num_data = mymodem.subcarrier_count*6
    data = np.random.rand(num_data)
    #data = np.random.rand(9996)
    mapped = mymodem.constellation.map(data)
    packs,mapped_in = mymodem.packetize_iq(mapped)
    
    p = packs[0]
    for i,p in enumerate(packs):
        p.S[21].freq_list+=27.5e9
        out_name = os.path.join(out_dir,'packet_{}.s1p'.format(i))
        #print(out_name)
        #p.write(out_name)
    #p.plot_iq(mymodem['constellation'])
    packs_out = []
    for p in packs:
        p.data+=complex(1,0)
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
        
        
        
        
        
        
        
        
        