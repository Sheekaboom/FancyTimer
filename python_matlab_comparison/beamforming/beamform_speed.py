# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:36:05 2020

@author: aweis
"""

from pycom.aoa.CBF import CBF
import numpy as np

def beamform_speed(num_angles,dtype='double'):
    '''
    @brief This function is used to calculate beamforming for given number of
        input angles per plane (az,el). Good for speed testing of languages.
    @note this is calculated for a 35 by 35 element equispaced planar array
       and includes time required to both synthesize the data for a single
       incident plane wave and calculate beamforming values at different
       angles.
    @param[in] num_angles - number of angles to calculate on each plane
       (az,el). The total number of angles will be num_angles^2 after meshgridding
    @param[in/OPT] dtype - string ('double' or 'single') for what dtypes to use
        can also be np.cdouble, or np.double
    @return [beamformed values, azimuth angles, elevation angles]
    '''
    dub_set = {'double',np.cdouble,np.double}
    if dtype in dub_set:
        dtype  = np.double
        cdtype = np.cdouble
    else:
        dtype  = np.single
        cdtype = np.csingle
        
    freqs = np.array([40e9],dtype=dtype); #frequency
    numel = [35,35,1]; #number of elements in x,y
    
    #now calculate everything
    spacing = 2.99e8/max(freqs)/2; #get our lambda/2
    [Yel,Xel,Zel] = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing); #create our positions
    pos = np.array([Xel.flatten(),Yel.flatten(),Zel.flatten()],dtype=dtype).T; #get our position [x,y,z] list
    
    #create all of our angles
    azi = np.deg2rad(np.linspace(-90,90,num_angles),dtype=dtype);
    eli = np.deg2rad(np.linspace(-90,90,num_angles),dtype=dtype);
    [AZ,EL] = np.meshgrid(azi,eli);
    
    #flatten the arrays for beamforming
    az = AZ.flatten()
    el = EL.flatten()
    
    #calculate weights and synthesize data
    weights = np.ones(pos.shape[0],dtype=cdtype); #get our weights
    sv = CBF.synthesize_data(freqs[0],pos,-np.pi/4,np.pi/4,dtype=cdtype);
    msv = sv;
    meas_vals = np.tile(msv,len(freqs));
    
    #get our beamformed values
    bf_vals = CBF.calculate(freqs,pos,meas_vals,az,el,weights=weights);
    
    #reshape our values for returning
    azr = np.reshape(az,AZ.shape);
    elr = np.reshape(el,AZ.shape);
    bfr = np.reshape(bf_vals,AZ.shape);
    
    return bfr,azr,elr


if __name__=='__main__':
    
    import plotly.graph_objects as go
    from pycom.base.OperationTimer import fancy_timeit 
    from WeissTools.python.PlotTools import format_plot
    
    dtype = np.cdouble
    
    #tstats = fancy_timeit(lambda: beamform_speed(500,dtype=dtype))
    
    #print(tstats['mean'])
    
    bfv,az,el = beamform_speed(181,dtype=dtype)
    fig = go.Figure(go.Surface(x=az,y=el,z=10*np.log10(np.abs(bfv))))
    fig = format_plot(fig,font_size=12)
    fig.write_html('../../docs/python_matlab_speed_testing/figs/beamform_results.html')
    fig.show()
    
    
    
    
    
    
