# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:40:46 2019

@author: ajw5
"""
from Modulation import Modem
from Modulation import ModulatedSignal,ModulatedPacket
from Modulation import lowpass_filter_zero_phase,bandstop_filter
from QAM import QAMConstellation
import numpy as np
import matplotlib.pyplot as plt

class OFDMModem(Modem):
    '''
    @brief class to generate OFDM signals in frequency and time domain
    '''
    def __init__(self,M,subcarrier_count,subcarrier_spacing,**arg_options):
        '''
        @brief constructor for the OFDMModem class
        @param[in] M - size of constellation (e.g. 16-QAM, M-QAM)
        @param[in] subcarrier_count - number of subcarrier channels
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
        self['subcarrier_count'] = subcarrier_count

        
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
        
    def packetize_iq(self,iq_points,**kwargs):
        '''
        @brief take a set of iq points (complex numbers) and split them into packets for each time period
            The length of each packet will be equal to the number of subcarriers
            Extra channels in last packet will be filled with 'padding' value
        @param[in] iq_points - list of iq points to split up
        @param[in] kwargs - keyword args as follows
            padding - complex number to pad unused channels with (when required)
        @return list of numpy arrays with each of the channel values
        '''
        options = {}
        options['padding'] = complex(0,0)
        for k,v in kwargs.items():
            options[k] = v
        num_pts = len(iq_points)
        iq_points = np.array(iq_points)
        mod_val = (len(iq_points)%self.subcarrier_count) #amount of padding required
        if mod_val is not 0:
            num_pad = self.subcarrier_count-mod_val
            print("Warning {} iq points not divisible by {} channels. Padding end with {}".format(num_pts,self.subcarrier_count,options['padding']))
            pad = np.full((num_pad,),options['padding'])
            iq_points = np.concatenate((iq_points,pad))
        packets = np.split(iq_points,iq_points.shape[0]/self.subcarrier_count)
        pack_list = []
        subcarriers = np.arange(self.subcarrier_count)*self.subcarrier_spacing
        for p in packets:
            pack_list.append(OFDMPacket(p,subcarriers))
        return pack_list
        
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
        @param[in] ofdm_signal -ofdm signal class with frequency domain packets
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
    def __init__(self,data,freqs,**kwargs):
        '''
        @brief constructor
        @param[in] data - complex modulated data on each subcarrier
        @param[in] freqs - subcarrier frequency list
        '''
        super().__init__(data,freqs,**kwargs)
        
if __name__=='__main__':
    
    import copy
    import os
    from Modulation import plot_frequency_domain
    
    out_dir = r'Q:\public\Quimby\Students\Alec\PhD\ofdm_tests\ofdm_packets_64QAM_1666SC'
    
    mymodem = OFDMModem(64,1666,60e3)
    data = 'testing'.encode()
    data = np.random.rand(9996)
    mapped = mymodem.constellation.map(data)
    packs = mymodem.packetize_iq(mapped)
    p = packs[0]
    for i,p in enumerate(packs):
        p.S[21].freq_list+=27.5e9
        out_name = os.path.join(out_dir,'packet_{}.s1p'.format(i))
        #print(out_name)
        p.write(out_name)
    p.plot_iq(mymodem['constellation'])
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
        
        
        
        
        
        
        
        
        