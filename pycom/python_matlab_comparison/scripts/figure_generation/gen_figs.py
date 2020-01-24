# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:23:52 2020
@brief generate timing figures with plotly and save to a variety of formats
@note this requires https://github.com/Sheekaboom/WeissTools
@author: aweis
"""

import plotly.graph_objects as go
import scipy.io as spio
import os
import re
import numpy as np
import copy

from pycom.base.OperationTimer import FancyTimerStatsSet
from WeissTools.python.PlotTools import save_plot,format_plot
from WeissTools.python.MatTools import load_mat_dict

def plot_timeit_sweep(stats,name,square_sizes=True):
    '''
    @brief plot a figure for a timeit sweep
    @param[in] stats - statstics to plot
    @param[in] name - name of the trace
    @param[in/OPT] square_sizes - whether or not to square size_m (default True)
    @return a plotly trace to add to a figure
    '''
    #first extract data into arrays
    stats = FancyTimerStatsSet(stats)
    data = stats.get_stats_as_arrays()
    if square_sizes:
        size_tot = data['size']**2
    else: #fft does not square
        size_tot = data['size']
    line = go.Scatter(x=size_tot,y=data['mean'],name=name,mode='lines+markers',
                      error_y = dict(type='data',array=data['stdev'],visible=True))
    return line

def plot_relative_timeit_sweep(stats,rel_stats,name,square_sizes=True):
    '''
    @brief Plot a figure for a timeit sweep relative to another value
    @param[in] stats - statistics to plot
    @param[in] name - name of the trace (for labeling)
    @cite https://www.nde-ed.org/GeneralResources/Uncertainty/Combined.htm
    '''
    #first extract data into arrays
    stats     = FancyTimerStatsSet(stats)
    rel_stats = FancyTimerStatsSet(rel_stats)
    data      = stats.get_stats_as_arrays()
    rel_data  = rel_stats.get_stats_as_arrays()
    if square_sizes:
        size_tot = data['size']**2
    else: #fft does not square
        size_tot = data['size']
    #this is to fix an issue with ssolve not including 1 for numpy
    has_idx = np.isin(rel_data['size'],data['size'])
    rel_mean = data['mean']/rel_data['mean'][has_idx]
    data_uncert_ratio = np.abs(data['stdev']/data['mean'])
    rdata_uncert_ratio = np.abs(rel_data['stdev']/rel_data['mean'])
    rel_stdev = np.sqrt(data_uncert_ratio**2+rdata_uncert_ratio[has_idx]**2)
    line = go.Scatter(x=size_tot,y=rel_mean,name=name,mode='lines+markers',
                      error_y = dict(type='data',array=rel_stdev,visible=True))
    return line

def gen_time_figs(data_dict,data_names):
    '''
    @brief generate figures from input data
    @param[in] data_dict - dictionary of data with keys as the labels
    @param[in] data_names - names of the data to extract (e.g. ['fft','sum'])
    '''
    fig_list = []
    for dname in data_names:
        if 'fft' in dname:
            sqsz = False
        else:
            sqsz = True
        fig = go.Figure()
        line_list = []
        for k,v in data_dict.items(): #loop through each value in data_dict
            if dname in v.keys(): #if it exists
                line_list.append(plot_timeit_sweep(v[dname], k,sqsz))
        #now add the traces
        fig.add_traces(line_list)
        fig.update_layout(yaxis_title = 'Runtime (s)',xaxis_title='Number of Elements')
        fig_list.append(fig)
    return fig_list

def gen_rel_time_figs(data_dict,data_names):
    '''
    @brief plot data dictionary relative to matlab data
    @note this assumes we have data_dict keys MATLAB Single and MATLAB Double
    '''
    #make a copy for popping
    data_dict = copy.deepcopy(data_dict)
    
    #get our values to get relative to
    rel_single = data_dict.pop('MATLAB Single',None)
    rel_double = data_dict.pop('MATLAB Double',None)
    
    fig_list = []
    for dname in data_names:
        if 'fft' in dname:
            sqsz = False
        else:
            sqsz = True
        #create the figure
        fig = go.Figure()
        #now lets plot single if the relative values exist
        line_list = []
        if rel_single is not None and dname in rel_single.keys(): #if we have single data
            for k,v in data_dict.items():
                if 'Single' in k and dname in v.keys(): #if its a single value
                    line_list.append(plot_relative_timeit_sweep(v[dname],
                                                            rel_single[dname],
                                                            k,sqsz))
        if rel_double is not None and dname in rel_double.keys(): #if we have single data
            for k,v in data_dict.items():
                if 'Double' in k and dname in v.keys(): #if its a double value
                    line_list.append(plot_relative_timeit_sweep(v[dname],
                                                            rel_double[dname],
                                                            k,sqsz))
        fig.add_traces(line_list)
        fig.update_layout(yaxis_title = 'Relative Runtime (s)',xaxis_title='Number of Elements')
        fig_list.append(fig)
    return fig_list
                    
                
if __name__=='__main__':    
            
    run_types = []
    data_names_list = []
    run_types.append('basic'); data_names_list.append(['add','sub','mult','div','exp','combo'])
    run_types.append('extended');data_names_list.append(['sum','fft'])
    run_types.append('matrix'); data_names_list.append(['lu','matmul','solve'])
    run_types.append('sparse'); data_names_list.append(['ssolve'])
    run_types.append('fdfd'); data_names_list.append(['fdfd'])
    run_types.append('beamforming'); data_names_list.append(['beamforming'])

    #lets get our different possible libraries
    lib_data_shorts = ['py_np','py_nb','py_sp','py_mf'  ]
    lib_data_names  = ['Numpy','Numba','Scipy','MKL_FFT']

    #plot each bit of data
    for run_type,data_names in zip(run_types,data_names_list):
        
        #data directory
        data_dir = './data'
        out_folder = './figs/plotly'
        
        data_dict_single = {}
        data_dict_double = {}
        
        #load and unpack matlab. This is done separately becaues the var names start with m_
        mat_data = load_mat_dict(os.path.join(data_dir,'stats_mat_{}.mat'.format(run_type)))
        data_dict_single['MATLAB'] = mat_data['m_single']
        data_dict_double['MATLAB'] = mat_data['m_double']
        
        #now see if we have the data for the specific library
        for lib_short,lib_name in zip(lib_data_shorts,lib_data_names):
            data_path = os.path.join(data_dir,'stats_{}_{}.mat'.format(lib_short,run_type))
            if os.path.exists(data_path): #then lets load if it exists
                cur_data = load_mat_dict(data_path)
                data_dict_single[lib_name] = cur_data['single']
                data_dict_double[lib_name] = cur_data['double']
        
        #now pack them together
        data_dict_single = {k+' Single':v for k,v in data_dict_single.items()}
        data_dict_double = {k+' Double':v for k,v in data_dict_double.items()}
        data_dict = data_dict_single; data_dict.update(data_dict_double)
        
        if run_type=='sparse': #if its fdfd remove matlab single (it doesnt exist)
            data_dict.pop('MATLAB Single')
            
        if run_type == 'fdfd':
            pass
        
        # now plot   
        fig_list = gen_time_figs(data_dict,data_names)
        fig_list[0].show()
        rel_fig_list = gen_rel_time_figs(data_dict,data_names)
        for name,fig in zip(data_names,fig_list):
            save_plot(fig,name,out_folder,verbose=True,margins={'t':20,'b':20,'l':20,'r':20})
            #format_plot(fig)
        for name,fig in zip(data_names,rel_fig_list):
            save_plot(fig,name+'_relative',out_folder,verbose=True,margins={'t':20,'b':20,'l':20,'r':20})
            #format_plot(fig)
        






