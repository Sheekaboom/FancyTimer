# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:21:00 2018

@author: ajw5
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

class QamSim:
    
    #number of symbols 'qam_syms' must be a power of two and have an integer square root (ie, 4,16,64,256)
    def __init__(self,qam_syms,bw=.5,fc = 1.,time_res=.01):
        self.qam_syms = qam_syms;
        self.const_x_size = int(np.sqrt(qam_syms));
        self.const_y_size = int(np.sqrt(qam_syms));
        self.const_x_range = 1;
        self.const_y_range = 1;
        #self.const_size   = const_x_size*const_y_size;
        self.fc = fc;
        self.bw = bw;
        self.oversample_ratio = 3; #demod window size is half our bandwidth
        self.window_time = 1/float(bw)
        self.time_step = .001e-9;#time_res/fc; #TODO Interpolate not hard code here
        self.window_times = np.arange(self.time_step,self.window_time,self.time_step);
        self.window_size = (len(self.window_times))
        
        #grid locations
        self.grid_x_arr = np.linspace(-self.const_x_range/2.,
                                      self.const_x_range/2.,self.const_x_size+1) 
        self.grid_y_arr = np.linspace(-self.const_y_range/2.,
                                      self.const_y_range/2.,self.const_y_size+1)
        self.center_x = self.grid_x_arr[0:-1]+np.diff(self.grid_x_arr)/2.;
        self.center_y = self.grid_y_arr[0:-1]+np.diff(self.grid_y_arr)/2.;
        self.build_constellation();
        
        self.phase_adj = 0;
        self.mag_adj   = 15;
        self.channel_delay_idx= 0; #delay of channel
        
        self.lowpass_fc = fc/2;
        self.lowpass_order = 3;
        
    #construct constellation and our mapping
    def build_constellation(self):
        #change to gray code mapping later
        self.const_map={}
        self.const_map_inv={}
        [x,y] = np.reshape(np.meshgrid(self.center_x,self.center_y),(2,-1))
        for i in range(len(x)):
            #print(x[i]);
            #print(y[i])
            self.const_map.update({i:(x[i],y[i])})
            self.const_map_inv.update({(x[i],y[i]):i})
            #place text on plot
    
    #map to our current constellation
    #give ints that fit in number of symbols
    def map_to_constellation(self,data_ints):
        #if(np.any(self.qam_syms<=i for i in data_ints)):
        #    print("Error: Some data out of range")
        #    return;
        vals =  [self.const_map[i] for i in data_ints];
        i_list,q_list = zip(*vals)
        return i_list,q_list
        
    def encode_qam_waveform_old(self,Ilist,Qlist,phase_adj=0,mag_adj=1):
        if(len(Ilist)!=len(Qlist)):
            print("ERROR: I and Q lists must be the same size");
            return -1;
        time = [0];#np.arange(self.time_step,self.window_time,self.time_step);
        iq_waveform =  [];#np.zeros(len(time));#self.encode_iq_point(Ilist[0],Qlist[0],time)
        for i in range(len(Ilist)):
            curTime = time[-1]+np.arange(self.time_step,self.window_time,self.time_step);
            cur_iq = self.encode_iq_point(Ilist[i],Qlist[i],curTime)
            iq_waveform = np.concatenate((iq_waveform,cur_iq,));
            time = np.concatenate((time,curTime,))
        return iq_waveform,time[1:];
    
    def encode_qam_waveform(self,Ilist,Qlist,phase_adj=0,mag_adj=1,do_lowpass=1):
        if(len(Ilist)!=len(Qlist)):
            print("ERROR: I and Q lists must be the same size");
            return -1;
        i_bb,q_bb,t_bb=self.encode_baseband(Ilist,Qlist,do_lowpass=do_lowpass);
        return self.mix_and_sum_iq(i_bb,q_bb,t_bb);
    
    def decode_qam_waveform_old(self,iq_waveform,times,phase_adj=0,mag_adj=1,num_symbols=-1,window_shift=0):
        if(num_symbols<0):
            num_symbols = int(len(iq_waveform)/self.window_size)-2;
        Ilist=[];Qlist=[];ws_list = [];we_list=[];
        for i in range(num_symbols):
            win_start = int((i*self.window_size)+(self.window_size/self.oversample_ratio)/2+self.window_size*window_shift);
            win_end   = int((win_start+(self.window_size/self.oversample_ratio)))
            iq_vals = iq_waveform[win_start:win_end];
            time_vals = times[win_start:win_end];
            I,Q = self.decode_iq_point(iq_vals,time_vals)
            Ilist.append(I);Qlist.append(Q)
            ws_list.append(times[win_start]);we_list.append(times[win_end]);
            
        Ilist,Qlist = self.adj_iq(Ilist,Qlist,phase_adj,mag_adj)    
        
        return Ilist,Qlist,ws_list,we_list        
    
    def decode_qam_waveform(self,iq_waveform,times,phase_adj=0,mag_adj=1,num_symbols=-1,do_lowpass=0,window_shift=0):
        i_bb,q_bb = self.mix_to_baseband(iq_waveform,times);
        return self.decode_baseband(i_bb,q_bb,times,phase_adj=phase_adj,mag_adj=mag_adj,num_symbols=num_symbols,do_lowpass=do_lowpass,window_shift=window_shift)
        
    #baseband iq and filter with lpf
    def encode_baseband(self,Ilist,Qlist,do_lowpass=1):
        #concatenate values and 
        time = [0];#np.arange(self.time_step,self.window_time,self.time_step);
        i_bb =  [];#np.zeros(len(time));#self.encode_iq_point(Ilist[0],Qlist[0],time)
        q_bb = [];
        for i in range(len(Ilist)):
            curTime = time[-1]+np.arange(self.time_step,self.window_time,self.time_step);
            i_val = np.ones(len(curTime))*Ilist[i];
            q_val = np.ones(len(curTime))*Qlist[i];
            i_bb = np.concatenate((i_bb,i_val,));
            q_bb = np.concatenate((q_bb,q_val,));
            time = np.concatenate((time,curTime,))
        if(do_lowpass):
            i_bb = self.lowpass_baseband(i_bb);
            q_bb = self.lowpass_baseband(q_bb);
        return i_bb,q_bb,time[1:]
            
    def lowpass_baseband(self,arr,cutoff_freq=-1):
        # from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
        nyq = 0.5/self.time_step;
        if(cutoff_freq<0):
            norm_cut = self.lowpass_fc/nyq;
        else:
            norm_cut = cutoff_freq/nyq;
        b,a = scipy.signal.butter(self.lowpass_order,norm_cut,btype='low',analog=False)
        out = scipy.signal.lfilter(b,a,arr);
        return out;
        
    def mix_and_sum_iq(self,i_bb,q_bb,time):
        times = np.array(time)
        i_bb = np.array(i_bb);
        q_bb = np.array(q_bb);
        II = i_bb*np.cos(2.*np.pi*self.fc*times);
        QQ = q_bb*np.sin(2.*np.pi*self.fc*times);
        IQ = II-QQ;
        return IQ,times;
    
    def mix_to_baseband(self,IQ,time):
        iq_in = np.array(IQ);
        times = np.array(time);
        i_bb =  iq_in*np.cos(np.pi*2.*self.fc*times);#/np.sum(np.square(np.cos(np.pi*2.*self.fc*times)))
        q_bb = -iq_in*np.sin(np.pi*2.*self.fc*times);#/np.sum(np.square(np.sin(np.pi*2.*self.fc*times)))
        return i_bb,q_bb;
    
    def decode_baseband(self,i_bb,q_bb,time,phase_adj=0,mag_adj=1,do_lowpass=1,num_symbols=-1,window_shift=0):
        times = np.array(time);
        if(do_lowpass):
            i_bb = self.lowpass_baseband(i_bb,cutoff_freq=self.lowpass_order*3.);
            q_bb = self.lowpass_baseband(q_bb,cutoff_freq=self.lowpass_order*3.);
        if(num_symbols<0):
            num_symbols = int(len(i_bb)/self.window_size)-2;
            
        i_list=[]; q_list=[]; ws_list=[]; we_list=[];
        for i in range(num_symbols):
            win_start = int((i*self.window_size)+(self.window_size/self.oversample_ratio)/2+self.window_size*window_shift);
            win_end   = int((win_start+(self.window_size/self.oversample_ratio)))
            i_vals = i_bb[win_start:win_end];
            q_vals = q_bb[win_start:win_end];
            time_vals = times[win_start:win_end];
            I = np.mean(i_vals)*2.
            Q = np.mean(q_vals)*2.
            i_list.append(I);q_list.append(Q)
            ws_list.append(times[win_start]);we_list.append(times[win_end]);
            
        i_list,q_list = self.adj_iq(i_list,q_list,phase_adj,mag_adj)    
        
        return i_list,q_list,ws_list,we_list     
        
    #iq over time array t
    def encode_iq_point(self,I,Q,times):  
        II = I*np.cos(2.*np.pi*self.fc*times);
        QQ = Q*np.sin(2.*np.pi*self.fc*times);
        IQ = II-QQ;
        return IQ;
    
    def decode_iq_point(self,IQ,times):
        I = ((np.sum(IQ*np.cos(np.pi*2.*self.fc*times)))/
             np.sum(np.square(np.cos(np.pi*2.*self.fc*times))))
        Q = -((np.sum(IQ*np.sin(np.pi*2.*self.fc*times)))/
              np.sum(np.square(np.sin(np.pi*2.*self.fc*times))))
        I = np.mean(IQ*np.cos(np.pi*2.*self.fc*times))*2
        Q = np.mean(IQ*np.sin(np.pi*2.*self.fc*times))*2
        
        return I,Q;
    
    #add phase in degrees and multiply magnitude
    def adj_iq(self,I,Q,phase,mag):
        i_old = np.array(I);
        q_old = np.array(Q);
        phase_vals = np.arctan2(q_old,i_old)*180./np.pi;
        mag_vals = np.sqrt(np.square(i_old)+np.square(q_old));
        mag_vals*=mag
        phase_vals+=phase
        i_new = mag_vals*np.cos(phase_vals*np.pi/180.)
        q_new = mag_vals*np.sin(phase_vals*np.pi/180.)
        return i_new,q_new
    
    def init_iq_plot(self):
        self.iq_fig = plt.figure();
        #build constellation grid outline
        plt.axvline(0,c='k');
        plt.axhline(0,c='k');
        for x in self.grid_y_arr:
            plt.axvline(x,ls='-.');
        for y in self.grid_y_arr:
            plt.axhline(y,ls='-.');
        #now draw center points
        [cpx,cpy] = np.meshgrid(self.center_x,self.center_y);
        x = np.reshape(cpx,(-1))
        y = np.reshape(cpy,(-1))
        for i in range(len(x)):
            plt.text(x[i],y[i]+(.02*self.const_y_range),str(i))
        plt.scatter(cpx,cpy)
        
            
    def plot_iq_points(self,I,Q,text_labels='none',colors='none'):
        plt.figure(self.iq_fig.number); #set as current figure
        if(colors=='none'):
            plt.scatter(I,Q)
        else:
            plt.scatter(I,Q,c=colors)       
        if(text_labels!='none'):
            for i in range(len(text_labels)):
                plt.text(I[i],Q[i],text_labels[i])
    
    #get the nearest value. Along with the distance from the constellation
    #check and see if we unmapped to correct key
    def unmap_from_constellation(self,I,Q,key_vals):
        #go through each IQ point and find the nearest constellation point
        ia = np.array(I);
        qa = np.array(Q);
        #get what grid we are inside
        i_idx = np.searchsorted(self.grid_x_arr[1:-1],ia);
        q_idx = np.searchsorted(self.grid_y_arr[1:-1],qa);
        #get the distance from the centers and the centers
        i_cent=[self.center_x[i_idx[i]] for i in range(len(i_idx))];
        q_cent=[self.center_y[q_idx[i]] for i in range(len(q_idx))];
        i_dist=[i_cent[i]-ia[i] for i in range(len(i_idx))];
        q_dist=[q_cent[i]-qa[i] for i in range(len(q_idx))];
        vals  =[self.const_map_inv[(i_cent[i],q_cent[i])] for i in range(len(i_idx))];
        phase_diff = [];
        mag_diff   = [];
        for i in range(len(i_idx)):
            #get the nearest constellation value
            #now get our phase offset
            desired_phase = np.arctan2(q_cent[i],i_cent[i])*180./np.pi;
            actual_phase     = np.arctan2(qa[i],ia[i])*180./np.pi;
            phase_diff.append(desired_phase-actual_phase);
            #and magnitude offset
            desired_mag = np.sqrt(np.square(i_cent[i])+np.square(q_cent[i]));
            actual_mag = np.sqrt(np.square(ia[i])+np.square(qa[i]));
            mag_diff.append(desired_mag-actual_mag);
        
        #distance from the place its supposed to be (given a key)
        const_vals = [self.const_map[k] for k in key_vals]
        i_true,q_true = zip(*const_vals)
        vm_real= np.sqrt(np.square(ia-i_true)+np.square(qa-q_true));
        #get evm distance for each point' (vector magnitude)
        vm_list = np.sqrt(np.square(ia-i_cent)+np.square(qa-q_cent));
        p_evm = np.mean(vm_list)
        p_evmr = np.mean(vm_real)
        #p_evm   = np.square(np.mean(ia-i_cent))+np.square(np.mean(qa-q_cent))
        p_avg   = np.mean(np.sqrt(np.square(i_cent)+np.square(q_cent)));
        p_avgr  = np.mean(np.sqrt(np.square(i_true)+np.square(q_true)));
        #evm     = np.sqrt(p_evm/p_avg)*100.;
        #evm     = 10.*np.log10(p_evm/p_avg)
        evm     = np.sqrt(p_evmr/p_avgr)*100.;
        #evm     = 10.*np.log10(p_evmr/p_avgr)
        #whether we mapped to the correct key
        correct_key = np.array(np.array(vals)==np.array(key_vals),dtype='int32')
            
        return vals,p_evm,p_avg,evm,phase_diff,mag_diff,correct_key;
    
    #test distance from known constellation point
    def find_correction_with_key(self,i_list,q_list,key):
        mag_diff=[];
        phase_diff=[];
        #get keys
        const_vals = [self.const_map[k] for k in key]
        i_true,q_true = zip(*const_vals)
        phase_rt = np.arctan2(q_true,i_true);
        phase_true = (phase_rt)*180./np.pi
        mag_true   = np.sqrt(np.square(i_true)+np.square(q_true));
        phase_rc = np.arctan2(q_list,i_list);
        phase_calc = (phase_rc)*180./np.pi
        mag_calc   = np.sqrt(np.square(i_list)+np.square(q_list));
        
        mag_diff = mag_true/mag_calc;
        phase_diff = phase_true-phase_calc;
        phase_diff += (phase_diff<0)*360.;
        phase_diff = np.unwrap(phase_diff*np.pi/180.)*180./np.pi;
        return phase_diff,mag_diff

    
    #find the delay of the channel
    def get_channel_delay(self,chan_imp_resp,key_len=10):
        random.seed(4567)
        possible_vals = range(self.qam_syms)
        send_key = [random.choice(possible_vals) for i in range(key_len)]
        il,ql = self.map_to_constellation(send_key)
        test_wf,t = self.encode_qam_waveform(il,ql)
        #find the maximum from the impulse response
        test_wfc = np.convolve(test_wf,chan_imp_resp,'full')
        sig_len = len(test_wf);
        test_corr = np.correlate(test_wfc,test_wf,'full');
        buff = round(self.window_size/4);
        start_idx = np.argmax(test_corr)-sig_len+buff;
        end_idx   = np.argmax(test_corr)+sig_len+buff;
        self.channel_delay_idx = start_idx
        return start_idx,end_idx;
        
    def get_channel_correction(self,chan_imp_resp,key_len=10,use_buff=1,do_lowpass=1):
        self.get_channel_delay(chan_imp_resp);
        random.seed(123)
        possible_vals = range(self.qam_syms)
        send_key = [random.choice(possible_vals) for i in range(key_len)]
        if(use_buff):
            send_key = self.add_buffer(send_key)
        i_list,q_list = self.map_to_constellation(send_key)
        wf_in,t_in = self.encode_qam_waveform(i_list,q_list,do_lowpass=do_lowpass)
        wf_out = np.convolve(wf_in,chan_imp_resp,'full')
        wf_out = wf_out[self.channel_delay_idx:] #adjust for time delay
        out_times = np.arange(len(wf_out))*self.time_step;
        out_i,out_q,_,_ =   self.decode_qam_waveform(wf_out,out_times,num_symbols=len(send_key))
        
        #assume we adjusted for channel offset already
        key_len = len(send_key)
        out_i = out_i[0:key_len]
        out_q = out_q[0:key_len]
        if(use_buff):
            send_key = self.strip_buffer(send_key)
            out_i    = self.strip_buffer(out_i)
            out_q    = self.strip_buffer(out_q)
        phase_offset,mag_offset = self.find_correction_with_key(out_i,out_q,send_key)
        
        self.phase_adj = phase_offset;
        self.mag_adj   = mag_offset;
        return phase_offset,mag_offset,send_key
    
    #add a buffer to fix convolution edge cases
    def add_buffer(self,arr,buf_size=5):
        possible_vals = range(self.qam_syms)
        stt_buff = [random.choice(possible_vals) for i in range(buf_size)]
        end_buff = [random.choice(possible_vals) for i in range(buf_size)]
        out_arr = stt_buff+arr+end_buff;
        return out_arr;
        
    def strip_buffer(self,arr,buf_size=5):
        out_arr = arr[buf_size:-buf_size]
        return out_arr;
        
    
        
def gen_root_raised_cosine(beta,Ts,times):
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
     
        
#err = []
#errI = []
#errQ = []
#times = np.arange(.1,3,.1)
#for tr in times:
#    qs = QAM_sim(2,2,t=np.arange(0,tr,0.01))
#    myI = 1;
#    myQ = 0;
#    IQ=qs.encode_iq_point(myI,myQ)
#    [I,Q] = qs.decode_iq_point(IQ)
#    errI.append(np.abs(I-myI));
#    errQ.append(np.abs(Q-myQ));
#    
#    locErr = np.sqrt(np.square(I-myI)+np.square(Q-myQ))
#    err.append(locErr)

import matplotlib.pyplot as plt
t = np.arange(-10,11,0.1)
ts = 1
betas = [0.001,0.01,0.1,0.5,0.7,1]
fig = plt.figure()
for b in betas:
    h = gen_root_raised_cosine(b,ts,t)
    plt.plot(t,h,label=r"$\beta={}$".format(b))
ax = plt.gca()
ax.legend()
    


from scipy import signal
if False:
    f = 2.4e9;
    bw = .2e9#.2*f;
    qam_syms = 16;
    numVals = 10;
    encode_lp = 1;
    decode_lp = 0;
    window_shft=0;
    
    
    txtname = './los_lin_pdp_full.txt';
    #txtname = './non_los_lin_pdp.txt';
    impResTime,impResVals = np.loadtxt(txtname,unpack=True);
    impResTime -= impResTime[0]
    imp_t_start = impResTime[0];
    imp_t_end   = impResTime[-1];
    impResTimeNew = np.arange(imp_t_start,imp_t_end,0.001)
    impResValsNew = np.interp(impResTimeNew,impResTime,impResVals);
    impResTime = impResTimeNew/1e9;
    impResVals = impResValsNew;
    #impResVals = signal.unit_impulse(len(impResTime),int(len(impResTime)/8))
    
    valList = range(qam_syms)
    
    random.seed(123)
    randVals = [random.choice(valList) for i in range(numVals)]
    rand_vals_str = [str(i) for i in randVals]
    
    print("Sim Started")
    qs = QamSim(qam_syms,bw=bw,fc=f);
    print("Correcting phase and magnitude for channel Response")
    phase_corr,mag_corr,test_key = qs.get_channel_correction(impResVals,do_lowpass=1,key_len=10)
    mag_adj = np.mean(mag_corr);#50000;
    phase_adj = np.mean(phase_corr)#;10;
    
    randVals = qs.add_buffer(randVals)
    
    print("Mapping to constellation")
    Il,Ql = qs.map_to_constellation(randVals)
    
    print("Encoding waveform")
    wf,t = qs.encode_qam_waveform(Il,Ql,do_lowpass=encode_lp)
    
    print("Running through channel (Convolve with impulse response)")
    wfc = np.convolve(wf,impResVals,'full') #fails here if impulse response in longer than waveform
    wfc = wfc[qs.channel_delay_idx:]
    tc = np.arange(len(wfc))*qs.time_step;
    
    print("Decoding waveform")
    ribb,rqbb = qs.mix_to_baseband(wfc,tc)
    outI,outQ,wsl,wel = qs.decode_qam_waveform(wfc,tc,phase_adj,mag_adj,num_symbols=len(randVals),do_lowpass=decode_lp,window_shift=window_shft)
    outI = qs.strip_buffer(outI)
    outQ = qs.strip_buffer(outQ)
    wsl = qs.strip_buffer(wsl)
    wel = qs.strip_buffer(wel)
    randVals = qs.strip_buffer(randVals);
    
    vals,p_avg,p_evm,evm,phase_off,mag_off,correct_key = qs.unmap_from_constellation(outI,outQ,randVals);
    #randVals  = qs.strip_buffer(randVals);
    
    print("Plotting");
    qs.init_iq_plot();
    qs.plot_iq_points(Il,Ql)
    #qs.plot_iq_points(np.multiply(outI,out_mult),np.multiply(outQ,out_mult))
    clist=['r','g','y','c']
    colors = [clist[i%len(clist)] for i in randVals]
    qs.plot_iq_points(outI,outQ,colors=colors)
    for i in range(len(outI)):
        plt.text(outI[i],outQ[i],rand_vals_str[i])
    plt.figure()
    plt.subplot(311)
    plt.plot(t,wf)
    plt.subplot(312)
    plt.plot(impResTime,impResVals)
    plt.subplot(313)
    plt.plot(tc,wfc)
    for i in range(len(wsl)):
        plt.axvline(wsl[i],c='r')
        plt.axvline(wel[i],c='k')
    tibb,tqbb,tt = qs.encode_baseband(Il,Ql);
    ribb,rqbb = qs.mix_to_baseband(wfc,tc);
    ribb=ribb[0:qs.window_size*numVals*2]
    rqbb=rqbb[0:qs.window_size*numVals*2]
    plt.figure();
    plt.subplot(211)
    plt.plot(tibb);plt.plot(tqbb);
    plt.subplot(212)
    plt.plot(ribb);plt.plot(rqbb);
    for i in range(len(wsl)):
        plt.axvline(wsl[i]/qs.time_step,c='r')
        plt.axvline(wel[i]/qs.time_step,c='k')

