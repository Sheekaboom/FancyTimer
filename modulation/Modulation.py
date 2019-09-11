# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:28:05 2019
Classes are labeled with numpy style, meethods are labeled with doxygen style
@author: aweis
"""

import numpy as np
import matplotlib.pyplot as plt
from samurai.base.SamuraiDict import SamuraiDict
from samurai.base.TouchstoneEditor import SnpEditor
from samurai.base.TouchstoneEditor import DEFAULT_HEADER as DEFAULT_SNP_HEADER

class Modem(SamuraiDict):
    '''
    This is a class for describing a modulation scheme. This will be inherited 
    by other modulation modules as subclasses.
   '''
    def __init__(self,**arg_options):
        '''
        @brief constructor to get options (and do other things)
        @param[in/OPT] arg_options - keyword arguments as follows
            sample_frequency - frequency for sample points
            carrier_frequency - frequency our qam will be upconverted to upon modulation
        '''
        super().__init__()
        self['sample_frequency']   = 200e9
        self['carrier_frequency'] = 20e9
        self['baud_rate']         = 100e6
        for k,v in arg_options.items():
            self[k] = v
        
   
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

    @property
    def options(self):
        '''
        @brief have this to be backward compatable with options dictionary
        '''
        return self

    def __getattr__(self,name):
        '''
        @brief check our options dictionary if an attirbute doesnt exist
        '''
        try: #return value from the dictionary
            rv = self[name]
            return rv
        except KeyError:
            raise AttributeError
    
class ModulatedSignal(SamuraiDict):
    '''
    @brief class to provide a template for a modulated signal
    This is simply a structure to hold all of the data describing the signal
    '''
    def __init__(self,data=None,**arg_options):
        '''
        @brief constructor to get options (and do other things)
        @param[in\OPT] data - data to be modulated
        @param[in/OPT] arg_options - keyword arguments as follows
            sample_frequency - frequency for sample points
            carrier_frequency - frequency our qam will be upconverted to upon modulation
        '''
        super().__init__()
        self['sample_frequency']   = 200e9
        self['carrier_frequency'] = 20e9
        self['baud_rate']         = 100e6
        self['packets']           = None
        self['type']              = None
        for k,v in arg_options.items():
            self[k] = v
        
        self.times = None #times for rf signal
        self.data = data
        self.data_type = type(data)
        self.baseband_dict = {} #dictionary for i and q
        self.rf_signal = None
    
    @property
    def bitstream(self):
        '''
        @brief get a bitstream of the data
        '''
        data_ba = bytearray(self.data)
        bits = np.unpackbits(data_ba)
        return bits

    @property
    def options(self):
        '''
        @brief have this to be backward compatable with options dictionary
        '''
        return self
    
    def plot_baseband(self):
        '''
        @brief plot all baseband data
        '''
        fig = plt.figure()
        all_baseband = []
        for key,val in self.baseband_dict.items():
            plt.plot(self.times,val,label="{} Baseband".format(key))
            all_baseband.append(val)
            
        plt_min = np.min(all_baseband)
        plt_max = np.max(all_baseband)
        plt.plot(self.times,self.clock*(plt_max-plt_min)+plt_min)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')
        ax.legend()
        return fig
   
    def plot_rf(self):
        '''
        @brief plot the upconverted rf signal
        '''
        plt.figure()
        plt.plot(self.times,self.rf_signal)
        return plt.gca()
    
    def apply_signal_to_snp_file(self,snp_path,out_path):
        '''
        @brief apply the modulated signal to an snp file. This will apply to all channels
        @param[in] snp_path - path to snp file to modulate
        @param[in] out_path - path to output file that has been modulated
        '''
        #first get the frequency domain data from our rf data
        freq_data = np.fft.fftshift(np.fft.fft(self.rf_signal))
        df = 1/self.times.max() #get frequency step
        N = len(self.times)
        freqs = np.array([df*n for n in range(-int(N/2),int(N/2)+1)])
       # freqs     = np.fft.fftfreq(len(freq_data),self.times[1]-self.times[0])
        from samurai.base.TouchstoneEditor import TouchstoneEditor #import snp editor
        mysnp = TouchstoneEditor(snp_path) #load the s2p file
        sfreqs = mysnp.S[21].freq_list*1e9 #get our s parameter frequencies assume GHZ (not great)
        angle_data = np.angle(freq_data)
        mag_data = np.abs(freq_data)
        freq_angle_interp = np.interp(sfreqs,freqs,angle_data)
        freq_mag_interp = np.interp(sfreqs,freqs,mag_data)
        freq_data_interp = freq_mag_interp*np.exp(1j*freq_angle_interp)
        for k in mysnp.S.keys():
            mysnp.S[k].raw *= freq_data_interp
        mysnp.write(out_path)
        return sfreqs,freq_data_interp,freqs,freq_data
    
    def apply_signal_to_samurai_metafile(self,metafile_path,out_dir):
        '''
        @brief apply the rf portion of the signal to a metafile for SAMURAI project
            this will go through all of the s-parameters in the metafile, 
            multiply the frequency domain version of the modulated signal by the s-param values,
            and rewrite out a new metafile, and a new set of s-parameter files with the modulated signal
        @param[in] metafile_path - path to the metafile to apply to
        @param[in] out_dir - output directory to save the new measurements and metafile to 
        '''
        pass
        
    def load_signal_from_snp_file(self,snp_path,load_key=21):
        '''
        @brief load an rf signal from a snp file and convert it to a time domain waveform.
            this will overwrite self.rf_signal. self.times MUST be set previous to this call
        @param[in] snp_path - path of the s2p file to load
        @param[in/OPT] load_key - what parameter to load the data from (e.g. S21. this defaults to 21)
        '''
        from samurai.base.TouchstoneEditor import SnpEditor #import snp editor
        #first load the snp file
        mysnp = SnpEditor(snp_path)
        sfreqs = mysnp.S[load_key].freq_list*1e9
        sdata = mysnp.S[load_key].raw #raw frequency data
        time_sdata = np.fft.ifft(sdata)
        dt = (1/np.diff(sfreqs).mean())/len(time_sdata)
        stimes = np.arange(len(time_sdata))*dt
        rf_data = np.interp(self.times,stimes,time_sdata)
        self.rf_signal = rf_data
        return self.times,self.rf_signal
        
    
def generate_gray_code_mapping(num_codes,constellation_function):
    '''
    @brief generate gray codes for a given number of codes
    @param[in] num_codes - the number of codes to generate (e.g. 256 is 0-255)
    @param[in] constellation_function - function to generate constellation locations
                for each code. The input to this will be the number of codes and the current code number.
                the return should be a complex number of where the point is on the real imaginary plane
    @return a dictionary mapping an array of bits to a constellation locations
    '''
    num_bits = get_num_bits(num_codes-1)
    codes = []
    constellation_dict = {}
    for i in range(num_codes): #for each desired code
        cur_const = constellation_function(i,num_codes)#generate our locations for iq here
        cur_code = np.zeros(num_bits) #flip to get MSB first in array
        for bn in range(num_bits):
            cur_code[bn] = (((i+2**bn)>>(bn+1)))%2
        cur_code = np.flip(cur_code,axis=0)
        codes.append(cur_code)
        constellation_dict[tuple(cur_code)] = cur_const
    return constellation_dict
   
def get_num_bits(number):
    '''
    @brief find the number of bits required for a number
    '''
    num_bits = np.ceil(np.log2(number)).astype('int') #much easier way
    #myba   = bytearray(np.array(number).byteswap().tobytes())
    #mybits = np.unpackbits(myba)
    #ones_loc = np.where(mybits==1)[0] #location of ones
    #if ones_loc.shape[0] == 0: #if the list is empty
    #    return 0 #if the number is 0 we have 0 bits
    #num_bits = mybits.shape[0] - ones_loc[0]
    return num_bits
   
def generate_root_raised_cosine_v1(beta,Ts,times):
    '''
    @brief generate a raised root cosine filter (from wikipedia equations)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    h = np.zeros(times.shape)
    for i,t in enumerate(times): #go through each time and calculate
        if t is 0:
            h[i] = 1/Ts*(1+beta(4/np.pi - 1))
        elif t is Ts/(4*beta) or t is -Ts/(4*beta):
            h[i] = beta/(Ts*np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            h[i] = 1/Ts*(np.sin(np.pi*(t/Ts)*(1-beta))+4*beta*(t/Ts)*np.cos(np.pi*(t/Ts)*(1+beta)))/(np.pi*(t/Ts)*(1-(4*beta*(t/Ts))**2))
    return h
    #right now runs saved values in
    #def run_impulse_response()
    
def generate_raised_cosine(beta,Ts,times):
    '''
    @brief generate a raised cosine filter (from  https://dspguru.com/dsp/reference/raised-cosine-and-root-raised-cosine-formulas/)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    sin_term = np.sin(np.pi*times/Ts)/(np.pi*times)
    cos_term = np.cos(np.pi*times*beta/Ts)/(1-4*beta**2*times**2/Ts**2)
    h = sin_term*cos_term
    return h
    #right now runs saved values in
    #def run_impulse_response()
    
def generate_root_raised_cosine(beta,Ts,times):
    '''
    @brief generate a raised root cosine filter (from  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120008631.pdf)
    @param[in] beta - rolloff factor 
    @param[in] Ts   - symbol period
    @param[in] times - times at which to evaluate the filter
    '''
    times = np.array(times)
    term_1 = 2*beta/(np.pi*np.sqrt(Ts))
    term_2_numerator = np.cos((1+beta)*np.pi*times/Ts)+np.sin((1-beta)*np.pi*times/Ts)/(4*beta*times/Ts)
    term_2_denominator = 1-(4*beta*times/Ts)**2
    h = term_1*term_2_numerator/term_2_denominator
    return h
    #right now runs saved values in
    #def run_impulse_response()


class ModulatedPacket(SnpEditor):
    '''
    @brief a class to save a single modulated packet. This could store time or frequency domain data but inherits from SnpEditor
        Which is technically frequency domain. To do time, simply treat the frequencies as times.
    '''
    def __init__(self,packet_data,time_freq,**kwargs):
        '''
        @brief constructor class for a packet
        @param[in] packet_data - iq packet data thats in the packet
        @param[in] time_freq - frequency or time ('x' axis) for packet_data
        @param[in/OPT] kwargs - keywrod arguments passed to SnpEditor init 
        '''
        super().__init__([1,time_freq],**kwargs) #init empty value
        self.options['header'] = DEFAULT_SNP_HEADER
        self.v1.raw = packet_data #set the packet data
        
    def _gen_dict_keys(self):
        return [21]
    
    def plot_iq(self,constellation):
        '''
        @brief plot our iq data on top of a specified QAMConstellation object
        @param[in] constellation - QAMConstellation object to plot the constellation.
        @note CURRENTLY ONLY WORKS WITH MATPLOTLIB
        '''
        import matplotlib.pyplot as plt
        constellation.plot()
        i = self.v1.raw.real
        q = self.v1.raw.imag
        plt.plot(i,q)
        
        

    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    t = np.arange(-10,11,0.1)
    ts = 1
    betas = [0.001,0.01,0.1,0.5,0.7,1]
    fig = plt.figure()
    for b in betas:
        h = generate_root_raised_cosine(b,ts,t)
        plt.plot(t,h,label=r"$\beta={}$".format(b))
    ax = plt.gca()
    ax.legend()
    
    times = np.arange(0,100,0.1)
    dirac = np.zeros(times.shape)
    dirac[int(times.shape[0]/2)] = 1
    plt.figure()
    plt.plot(times,dirac)
    dirac_conv = np.convolve(dirac,h,'same')
    plt.plot(times,dirac_conv)
    dirac_conv_conv = np.convolve(dirac_conv,h,'same')
    plt.plot(times,dirac_conv_conv)  

import scipy.signal
def lowpass_filter(data,time_step,cutoff_freq,order=5):
        # from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
        nyq = 0.5/time_step;
        norm_cut = cutoff_freq/nyq;
        b,a = scipy.signal.butter(order,norm_cut,btype='low',analog=False)
        out = scipy.signal.lfilter(b,a,data);
        return out;
    
def lowpass_filter_zero_phase(data,time_step,cutoff_freq,order=5):
        # from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    nyq = 0.5/time_step;
    norm_cut = cutoff_freq/nyq;
    b,a = scipy.signal.butter(order,norm_cut,btype='low',analog=False)
    out = scipy.signal.filtfilt(b,a,data);
    return out;

def bandstop_filter(data,time_step,cutoff_freq,q_factor=20):
    nyq = 0.5/time_step;
    norm_cut = cutoff_freq/nyq;
    b,a = scipy.signal.iirnotch(norm_cut,q_factor)
    out = scipy.signal.filtfilt(b,a,data);
    return out;

def plot_frequency_domain(data,dt):
    f_data = np.fft.fft(data)
    f_freqs= np.fft.fftfreq(len(f_data),dt)
    fig = plt.figure()
    plt.plot(f_freqs,f_data)
    return fig

#naive fourier series calculations (non-uniform foureier transform)
def dft(data,times,freqs_out):
    '''
    @brief calculate the fouerier series for nonuniform points
    @param[in] data - f(t) values for the corresponding times
    @param[in] times - times that correspond to the data points
    @param[in] freqs_out - list of output frequencies to calculate for
    '''
    #ran out of memory for vectorized computation
    freq_vals = []
    print("%5d of %5d" %(0,len(freqs_out)),end='')
    for i,f in enumerate(freqs_out):
        print('\b'*14+"%5d of %5d" %(i+1,len(freqs_out)),end='')
        data  = np.array(data).flatten()
        times = np.array(times).flatten()
        exp_val = np.exp(-1j*2*np.pi*f*times)
        fv = (data*exp_val).sum()
        freq_vals.append(fv)
    print('')
    return np.array(freq_vals)

def idft(data,freqs,times_out):
    '''
    @brief calculate the inverser fourier series at nonuniform points
    @param[in] data - F(f) values at corresponding frequencies
    @param[in] freqs - corresponding frequencies to data
    @param[in] times_out - times to calculate for
    '''
    times_out = -times_out #change -1j to 1j by dong this
    time_vals = dft(data,freqs,times_out)
    return time_vals














