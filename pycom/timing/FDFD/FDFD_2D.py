# -*- coding: utf-8 -*-

import numpy
np = numpy
import scipy.sparse
from scipy.sparse.linalg import spsolve as scispsolve

try:
    import cupy
    import cupyx.scipy.sparse
    from cupyx.scipy.sparse.linalg import lsqr as culsqr
except ModuleNotFoundError: #if cupy doesnt exist set it to none
    cupy = None
    

#%% First lets define some values needed to build#
#  the grid      (changed by user)              #
def FDFD_2D(num_cells_x=None,num_cells_y=None,dtype=np.cdouble,use_gpu=False):
    '''
    @brief Finite Difference Frequency Domain Solver for a cylindrical Scatterer.
    @date Fall 2017
    @author Alec Weiss
    @param[in/OPT] num_cells_x - number of cells to use in the x direction
    @param[in/OPT] num_cells_y - number of cells to use in the y direction
    @param[in/OPT] dtype - what dtype to use for our arrays (default cdouble)
    @param[in/OPT] use_gpu - wheter or not to use the GPU (default false)
    '''
    
    #%% use the gpu if desired
    a = 2
    if use_gpu:
        if cupy is not None: #then set the libraries accordingly
            sparse = cupyx.scipy.sparse
            np = cupy
            spsolve = lambda A,b: culsqr(A,b)[0]
        else:
            raise ModuleNotFoundError("Cupy or Cupyx not imported correctly")
    else:
        np = numpy
        sparse = scipy.sparse
        spsolve = scispsolve
        
    #dtype = np.cdouble
    
    #set sizes of things in our domain (in meters)
    cyl_sz_r = 10e-2; #radius of cylinder
    cyl_sz_x = 15e-2; #size of cylinder in x direction
    cyl_sz_y = 10e-2; #size of cylinder in y direction
    air_buffer_sz = 15e-2;#size of surrounding air buffer
    pml_thickness = 10e-2;# size of PML regions
    
    #set the number of cells in our grid (for testing vs size)
    #resolution will automatically be set from this
    #set the resolution of our grid in meters
    dx = 5e-3;dy = 5e-3;
    if num_cells_x is not None:
        size_x = cyl_sz_x+2*air_buffer_sz+2*pml_thickness
        dx = size_x/num_cells_x
    if num_cells_y is not None:
        size_y = cyl_sz_y+2*air_buffer_sz+2*pml_thickness
        dy = size_y/num_cells_y
    
    #our permitivities and permeabilities
    #of our cylinder
    eps_cyl = 1e10;
    mu_cyl  = 1;
    eps_0   = 8.85418782e-12;
    mu_0    = 1.25663706e-6;
    eps0    = eps_0;
    mu0     = mu_0;
    
    #our operation frequency
    freq = 3e9;
    omega = 2*np.pi*freq;
    inc_amp = 1; #incident wave amplitude
    
    #calculate our wavelength and wavenumber
    c0      = 2.99792458e8;
    lam = c0/freq;
    kx     = (2*np.pi)/lam;
    
    #%% Now lets set up our problem space variables  #
    
    #size in cells of each of our regions
    #size of our cylinders in each dim
    cyl_cells_x = np.int(np.ceil(cyl_sz_x/dx));
    cyl_cells_y = np.int(np.ceil(cyl_sz_y/dy));
    #size of our air buffer in cells
    airbuf_cells_x = np.int(np.ceil(air_buffer_sz/dx));
    airbuf_cells_y = np.int(np.ceil(air_buffer_sz/dy));
    #now pml region size in cells
    pml_cells_x = np.int(np.ceil(pml_thickness/dx));
    pml_cells_y = np.int(np.ceil(pml_thickness/dy));
    
    #total size of our grid total grid
    #just add up all of our pieces
    #*2 by pml and airbuf because one on each side
    cells_x = np.int((cyl_cells_x + 
        2*airbuf_cells_x + 2*pml_cells_x));
    cells_y = np.int((cyl_cells_y + 
        2*airbuf_cells_y + 2*pml_cells_y));
    
    
    #%% We can now begin building our cofficient matrices #
    
    #these matrices will later be translated into a sparse
    #matrix and then be cleared from memory
    #this just ensures values are set correctly and allows
    #for faster (than looping) assignment to sparse matrix
    
    #lets first create matrices from which we create other 
    #values THESE WILL BE CLEARED after usage
    eps_mat = np.ones((cells_x,cells_y),dtype=dtype)*eps0;
    mu_mat  = np.ones((cells_x,cells_y),dtype=dtype)*mu0;
    
    #initialize these to zeroes so
    #we can multiply to get values easily
    #get our maximum conductivity
    sig_max      = .3; #maximum conductivity
    n            = 2;
    sigma_ex_mat = np.zeros((cells_x,cells_y),dtype=dtype);
    sigma_ey_mat = np.zeros((cells_x,cells_y),dtype=dtype);
    sigma_mx_mat = np.zeros((cells_x,cells_y),dtype=dtype);
    sigma_my_mat = np.zeros((cells_x,cells_y),dtype=dtype);
    
    #now we fill our sigma matrices
    #use a parabolic conductivity
    hxe = (np.arange(1,pml_cells_x+1)).reshape((-1,1))* np.ones((pml_cells_x,cells_y)); #end h value
    hxs = (np.arange(pml_cells_x,0,-1)).reshape((-1,1))*np.ones((pml_cells_x,cells_y)); #start h value
    hye = (np.arange(1,pml_cells_y+1)).reshape((1,-1))*np.ones((cells_x,pml_cells_y));
    hys = (np.arange(pml_cells_y,0,-1)).reshape((1,-1))*np.ones((cells_x,pml_cells_y));
    sigma_ex_mat[0:pml_cells_x,:] = (sig_max*
        (hxs/pml_cells_x)**(n+1));
    sigma_ex_mat[sigma_ex_mat.shape[0]-pml_cells_x:,:] = (sig_max*
        (hxe/pml_cells_x)**(n+1));
    sigma_ey_mat[:,0:pml_cells_y] = (sig_max*
        (hys/pml_cells_y)**(n+1));
    sigma_ey_mat[:,sigma_ey_mat.shape[1]-pml_cells_y:] = (sig_max*
        (hye/pml_cells_y)**(n+1));
    
    sigma_mx_mat[0:pml_cells_x,:] = (sig_max*
        ((hxs+.5)/pml_cells_x)**(n+1)*mu0/eps0);
    sigma_mx_mat[sigma_mx_mat.shape[0]-pml_cells_x:,:] = (sig_max*
        ((hxe+.5)/pml_cells_x)**(n+1)*mu0/eps0);
    sigma_my_mat[:,0:pml_cells_y] = (sig_max*
        ((hys+.5)/pml_cells_y)**(n+1)*mu0/eps0);
    sigma_my_mat[:,sigma_my_mat.shape[1]-pml_cells_y:] = (sig_max*
        ((hye+.5)/pml_cells_y)**(n+1)*mu0/eps0);
    #Now we set our geometry values
    
    #CIRCULAR
    cells_r = np.ceil(cyl_sz_r/dx);
    	#circular middle
    icx = np.ceil(cells_x/2); #the center of the grid
    icy = np.ceil(cells_y/2);
    [CY,CX] = np.meshgrid(np.arange(1,cells_y+1),np.arange(1,cells_x+1));
    cyl_ind = np.sqrt((CX-icx)**2+(CY-icy)**2)<=cells_r;
    
    #first set our inner cylinder values 
    eps_mat[cyl_ind] = eps_cyl*eps_mat[cyl_ind];
    mu_mat[cyl_ind]  = mu_cyl*mu_mat[cyl_ind];
    
    #now we can adjust the PML values for epsilon
    eps_zx_mat = eps_mat + (sigma_ex_mat/(1j*omega));
    eps_zy_mat = eps_mat + (sigma_ey_mat/(1j*omega));
    eps_zi_mat = eps_mat;
    #now lets do the same thing for mu vals
    mu_xy_mat = mu_mat + (sigma_my_mat/(1j*omega));
    mu_yx_mat = mu_mat + (sigma_mx_mat/(1j*omega));
    mu_xi_mat = mu_mat;
    mu_yi_mat = mu_mat;
    
    #CHECKPOINT: WORKS TO HERE
    
    
    #%% Build our sources                                 #
    
    #source Ez = Ae^(-j*k_x*x)
    #Hy and Hx solved from Ez
    #mesh in x direction used for plane wave
    eta_x = np.sqrt(mu_xi_mat/eps_zi_mat);
    eta_y = np.sqrt(mu_yi_mat/eps_zi_mat);
    x_mesh = ((((np.arange(1,cells_x+1))*dx)-(np.ceil(cells_x/2)*dx)).reshape((-1,1))*
    									np.ones((cells_x,cells_y)));
    Ezi = inc_amp*np.exp(-1j*kx*x_mesh);
    Hxi = Ezi*1/eta_x;
    Hyi = Ezi*1/eta_y;
        
    #CHECKPOINT: WORKS TO HERE
    
    
    #%% Create our coefficient matrices                   #
    
    #these will be reshaped to fill our matrix A
    a = 1/(eps_zx_mat*mu_yx_mat*omega**2*dx**2);
    b = np.zeros_like(a);
    b[1:,:] = (1/(eps_zx_mat[1:,:]*
        mu_yx_mat[0:-1,:]*omega**2*dx**2));
    
    c = 1/(eps_zy_mat*mu_xy_mat*omega**2*dx**2);
    d = np.zeros_like(c);
    d[:,1:] = (1/(eps_zy_mat[:,1:]*
        mu_xy_mat[:,0:-1]*omega**2*dx**2));
    
    e = 1-(a+b+c+d);
    #have to use 2 to end here for -1 values
    f = np.zeros_like(e);
    f[1:,1:] = ((eps0-eps_zi_mat[1:,1:])/
            (eps_zi_mat[1:,1:])*Ezi[1:,1:] + 
        (mu0-mu_yi_mat[1:,1:])/
            (1j*omega*dx*eps_zx_mat[1:,1:]
            *mu_yi_mat[1:,1:])*Hyi[1:,1:] - 
        (mu0-mu_yi_mat[1:,0:-1])/
            (1j*omega*dx*eps_zx_mat[1:,1:]
            *mu_yi_mat[1:,0:-1])*Hyi[1:,0:-1] - 
        (mu0-mu_xi_mat[1:,1:])/
            (1j*omega*dy*eps_zy_mat[1:,1:]
            *mu_xi_mat[1:,1:])*Hxi[1:,1:] + 
        (mu0-mu_xi_mat[0:-1,1:])/
            (1j*omega*dy*eps_zy_mat[1:,1:]
            *mu_xi_mat[0:-1,1:])*Hxi[0:-1,1:]);
    
    #now we can clear all of our intermediate matrices
    #clear eps_zx_mat eps_zy_mat eps_zi_mat
    #clear mu_xy_mat mu_yx_mat mu_xi_mat mu_yi_mat
    
    # Setting up our final matrices including the sparse matrix
    #now lets build our final solvable matrices and vectors
    #now we will reshape all of our other coefficients
    #ar = np.reshape(a,(-1,1)); #(i+1,j)
    #br = np.reshape(b,(-1,1)); #(i-1,j)
    #cr = np.reshape(c,(-1,1)); #(i,j+1)
    #dr = np.reshape(d,(-1,1)); #(i,j-1)
    #er = np.reshape(e,(-1,1)); #(i,j)
    ar = a.transpose().flatten()
    br = b.transpose().flatten(); #(i-1,j)
    cr = c.transpose().flatten(); #(i,j+1)
    dr = d.transpose().flatten(); #(i,j-1)
    er = e.transpose().flatten(); #(i,j)
    
    #these will be y in Ax=y
    fr = np.transpose(f).conj().reshape((-1,1)); #(i,j)
    
    #clear a b c d e f
    
    #lets declare the size of 1 dimension of our sparse mat
    sparse_n = cells_x*cells_y; 
    
    #now lets create the indices for each of our values
    n    = np.arange(0,sparse_n);     #use this to modulo for wrapping around
    yIdx = n;          				  #indices in python range
    cIdx = np.mod(n+cells_x,sparse_n); #(i,j+1)
    dIdx = np.mod(n-cells_x,sparse_n); #(i,j-1)
    aIdx = np.mod(n-1,sparse_n);       #(i+1,j)
    bIdx = np.mod(n+1,sparse_n);       #(i-1,j)
    eIdx = n;                       #(i,j)
    
    #concatenate for sparse builder
    sparse_vals  = np.concatenate([ar  ,br  ,cr  ,dr  ,er  ],axis=0);
    sparse_x_idx = np.concatenate([aIdx,bIdx,cIdx,dIdx,eIdx],axis=0);
    sparse_y_idx = np.concatenate([yIdx,yIdx,yIdx,yIdx,yIdx],axis=0);
    
    #dtype = np.csingle
    #now build the sparse matrix
    A = sparse.csr_matrix((sparse_vals,(sparse_y_idx,sparse_x_idx)));
    
    #solve for our scatterfield
    x = spsolve(A,fr);
    #x = splinalg.lsqr(A,fr)[0]
    
    #reshape back into a matrix
    E_scat = np.reshape(x,(cells_y,cells_x));
    
    #create our total field
    E_tot = E_scat.transpose()+np.conj(Ezi);
    return E_tot,E_scat
    #rv = 0;
    

#%% Plotting
if __name__=='__main__':
    num_cells = 120
    from pycom.timing.OperationTimer import fancy_timeit
    time_stats = fancy_timeit(lambda: FDFD_2D(num_cells,num_cells,np.cdouble),num_reps=5)
    
    E_tot,E_scat = FDFD_2D(num_cells,num_cells,np.csingle)

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
    '''
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    cells_x,cells_y = E_scat.shape
    dx = 5e-3;dy = 5e-3;
    
    xp = np.arange(cells_x)*dx;yp = np.arange(cells_y)*dy;
    [Y,X] = np.meshgrid(xp,yp);

    fig = make_subplots(rows=1,cols=3,
                    specs=[[{'type': 'surface'},{'type': 'surface'},{'type':'scatter'}]],
                    subplot_titles=("Scattered Field","Total Field"))
    
#fig = make_subplots(rows=1,cols=3,
#                   specs=[[{'type': 'scatter'},{'type': 'scatter'},{'type':'scatter'}]],
#                    subplot_titles=("Incident Field","Total Field"))

    Zi = np.abs(E_scat)
    Zt = np.abs(E_tot)
    #fig = go.Figure(data=[go.Surface(z=Zi)])
    #fig.add_trace(go.Surface(z=np.abs(Hyi)),row=1,col=1)
    #fig.add_trace(go.Surface(z=np.abs(Ezi)),row=1,col=2)
    fig.add_trace(go.Surface(z=Zi),row=1,col=1)
    fig.add_trace(go.Surface(z=Zt),row=1,col=2)
    #fig.add_trace(go.Scatter(x=np.arange(len(sparse_vals)),y=sparse_vals.real),row=1,col=1)
    #fig.add_trace(go.Scatter(x=np.arange(len(er)),y=er.real),row=1,col=2)
    #fig.add_trace(go.Scatter(x=np.arange(len(fr)),y=fr.flatten().real),row=1,col=3)
    #fig.write_html('../../docs/python_matlab_speed_testing/figs/FDFD_results.html')
    fig.show()
    '''

'''
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
Z = np.abs(f.transpose())
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

ax = fig.add_subplot(122, projection='3d')
Z = np.angle(f.transpose())
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''

