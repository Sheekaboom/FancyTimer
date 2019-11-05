# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:53:18 2019

@author: aweis
"""
<<<<<<< HEAD
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

=======

def sub2ind(size,x,y):
	'''
	@brief change x,y indices to linear locations
	'''
	
>>>>>>> refs/remotes/origin/master
#
#   Finite Difference Frequency Domain Solver   #
#                                               #
# Written Fall 2017 By Alec Weiss               #
#               For Intro to Computatinal EM    #
#for a TE mode of propogation                   #
#

<<<<<<< HEAD
#%% First lets define some values needed to build#
#  the grid      (changed by user)              #
def FDFD_2D():

    #set the resolution of our grid in meters
    dx = 5e-3;dy = 5e-3;
    #dtype = np.cdouble
=======
#-----------------------------------------------#
# First lets define some values needed to build#
#  the grid      (changed by user)              #
#-----------------------------------------------#
def FDFD_2D():
    from numpy import *
    #circ or rect
    shape = 'rect';
    
    #set the resolution of our grid in meters
    dx = 5e-3;dy = 5e-3;
    
>>>>>>> refs/remotes/origin/master
    
    #set sizes of things in our domain (in meters)
    cyl_sz_r = 10e-2; #radius of cylinder
    cyl_sz_x = 15e-2; #size of cylinder in x direction
    cyl_sz_y = 10e-2; #size of cylinder in y direction
    air_buffer_sz = 15e-2;#size of surrounding air buffer
    pml_thickness = 10e-2;# size of PML regions
    
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
<<<<<<< HEAD
    omega = 2*np.pi*freq;
=======
    omega = 2*pi*freq;
>>>>>>> refs/remotes/origin/master
    inc_amp = 1; #incident wave amplitude
    
    #calculate our wavelength and wavenumber
    c0      = 2.99792458e8;
    lam = c0/freq;
<<<<<<< HEAD
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
=======
    kx     = (2*pi)/lam;
    
    #-----------------------------------------------#
    # Now lets set up our problem space variables  #
    #-----------------------------------------------#
    #size in cells of each of our regions
    #size of our cylinders in each dim
    cyl_cells_x = np.ceil(cyl_sz_x/dx);
    cyl_cells_y = np.ceil(cyl_sz_y/dy);
    #size of our air buffer in cells
    airbuf_cells_x = np.ceil(air_buffer_sz/dx);
    airbuf_cells_y = np.ceil(air_buffer_sz/dy);
    #now pml region size in cells
    pml_cells_x = np.ceil(pml_thickness/dx);
    pml_cells_y = np.ceil(pml_thickness/dy);
>>>>>>> refs/remotes/origin/master
    
    #total size of our grid total grid
    #just add up all of our pieces
    #*2 by pml and airbuf because one on each side
<<<<<<< HEAD
    cells_x = np.int((cyl_cells_x + 
        2*airbuf_cells_x + 2*pml_cells_x));
    cells_y = np.int((cyl_cells_y + 
        2*airbuf_cells_y + 2*pml_cells_y));
    
    
    #%% We can now begin building our cofficient matrices #
    
=======
    cells_x = (cyl_cells_x + 
        2*airbuf_cells_x + 2*pml_cells_x);
    cells_y = (cyl_cells_y + 
        2*airbuf_cells_y + 2*pml_cells_y);
    
    #----------------------------------------------------#
    # We can now begin building our cofficient matrices #
    #----------------------------------------------------#
>>>>>>> refs/remotes/origin/master
    #these matrices will later be translated into a sparse
    #matrix and then be cleared from memory
    #this just ensures values are set correctly and allows
    #for faster (than looping) assignment to sparse matrix
<<<<<<< HEAD
    
=======
    #-------
>>>>>>> refs/remotes/origin/master
    #lets first create matrices from which we create other 
    #values THESE WILL BE CLEARED after usage
    eps_mat = np.ones((cells_x,cells_y))*eps0;
    mu_mat  = np.ones((cells_x,cells_y))*mu0;
    
    #initialize these to zeroes so
    #we can multiply to get values easily
    #get our maximum conductivity
    sig_max      = .3; #maximum conductivity
    n            = 2;
    sigma_ex_mat = np.zeros((cells_x,cells_y));
    sigma_ey_mat = np.zeros((cells_x,cells_y));
    sigma_mx_mat = np.zeros((cells_x,cells_y));
    sigma_my_mat = np.zeros((cells_x,cells_y));
    
    #now we fill our sigma matrices
    #use a parabolic conductivity
<<<<<<< HEAD
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
=======
    hxe = (np.arange(1,pml_cells+1)).transpose()* np.ones((pml_cells_x,cells_y)); #end h value
    hxs = (np.arange(pml_cells,0,-1)).transpose()*np.ones((pml_cells_x,cells_y)); #start h value
    hye = (np.arange(1,pml_cells+1) )*np.ones((cells_x,pml_cells_y));
    hys = (np.arange(pml_cells,0,-1))*np.ones((cells_x,pml_cells_y));
    sigma_ex_mat[0:pml_cells_x-1,:] = sig_max*
        (hxs/pml_cells_x)**(n+1);
    sigma_ex_mat[sigma_ex_mat.shape[0]-pml_cells_x:-1,:] = (sig_max*
        (hxe/pml_cells_x)**(n+1));
    sigma_ey_mat[:,0:pml_cells_y-1] = (sig_max*
        (hys/pml_cells_y)**(n+1));
    sigma_ey_mat[:,sigma_ey_mat.shape[1]-pml_cells_y:-1] = (sig_max*
        (hye/pml_cells_y)**(n+1));
    
    sigma_mx_mat[0:pml_cells_x-1,:] = (sig_max*
        ((hxs+.5)/pml_cells_x)**(n+1)*mu0/eps0);
    sigma_mx_mat(sigma_mx_mat.shape[0]-pml_cells_x:-1,:] = (sig_max*
        ((hxe+.5)/pml_cells_x)**(n+1)*mu0/eps0);
    sigma_my_mat[:,0:pml_cells_y-1] = (sig_max*
        ((hys+.5)/pml_cells_y)**(n+1)*mu0/eps0);
    sigma_my_mat[:,sigma_my_mat.shape[1]-pml_cells_y:-1] = (sig_max*
        ((hye+.5)/pml_cells_y)**(n+1)*mu0/eps0);
    #Now we set our geometry values
    #RECTANGULAR
    if(shape=='rect'):
		#set the inner cube values
		cyl_x_start = 1+pml_cells_x+airbuf_cells_x-1;
		cyl_x_end   = cyl_x_start+cyl_cells_x-1-1;
		cyl_y_start = 1+pml_cells_y+airbuf_cells_y-1;
		cyl_y_end   = cyl_y_start+cyl_cells_y-1-1;
		[cyl_x,cyl_y] = meshgrid(cyl_x_start:cyl_x_end,cyl_y_start:cyl_y_end);
		cyl_ind = sub2ind(size(eps_mat),cyl_x,cyl_y);

    
    #CIRCULAR
    if(shape=='circ'):
		cells_r = np.ceil(cyl_sz_r/dx);
		#circular middle
		icx = np.ceil(cells_x/2); #the center of the grid
		icy = np.ceil(cells_y/2);
		cyl_ind = sub2ind(size(eps_mat),icx,icy);
		for i in range(cells_x):
			for j in range(cells_y):
				if(np.sqrt((i-icx)**2+(j-icy)**2)<=cells_r):
					#find our conductor locations
					cyl_ind = [cyl_ind,sub2ind(size(eps_mat),i,j)];
    
    #first set our inner cylinder values 
    eps_mat(cyl_ind) = eps_cyl*eps_mat(cyl_ind);
    mu_mat(cyl_ind)  = mu_cyl*mu_mat(cyl_ind);
    
    #now we can adjust the PML values for epsilon
    eps_zx_mat = eps_mat + (sigma_ex_mat/(1i*omega));
    eps_zy_mat = eps_mat + (sigma_ey_mat/(1i*omega));
    eps_zi_mat = eps_mat;
    #now lets do the same thing for mu vals
    mu_xy_mat = mu_mat + (sigma_my_mat/(1i*omega));
    mu_yx_mat = mu_mat + (sigma_mx_mat/(1i*omega));
>>>>>>> refs/remotes/origin/master
    mu_xi_mat = mu_mat;
    mu_yi_mat = mu_mat;
    
    #CHECKPOINT: WORKS TO HERE
    
<<<<<<< HEAD
    
    #%% Build our sources                                 #
    
=======
    #----------------------------------------------------#
    # Build our sources                                 #
    #----------------------------------------------------#
>>>>>>> refs/remotes/origin/master
    #source Ez = Ae^(-j*k_x*x)
    #Hy and Hx solved from Ez
    #mesh in x direction used for plane wave
    eta_x = np.sqrt(mu_xi_mat/eps_zi_mat);
    eta_y = np.sqrt(mu_yi_mat/eps_zi_mat);
<<<<<<< HEAD
    x_mesh = ((((np.arange(1,cells_x+1))*dx)-(np.ceil(cells_x/2)*dx)).reshape((-1,1))*
    									np.ones((cells_x,cells_y)));
    Ezi = inc_amp*np.exp(-1j*kx*x_mesh);
=======
    x_mesh = ((((np.arange(1,cells_x+1))*dx)-(np.ceil(cells_x/2)*dx)).transpose()*
									np.ones(cells_x,cells_y));
    Ezi = inc_amp*np.exp(-1i*kx*x_mesh);
>>>>>>> refs/remotes/origin/master
    Hxi = Ezi*1/eta_x;
    Hyi = Ezi*1/eta_y;
        
    #CHECKPOINT: WORKS TO HERE
    
<<<<<<< HEAD
    
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
=======
    #----------------------------------------------------#
    # Create our coefficient matrices                   #
    #----------------------------------------------------#
    #these will be reshaped to fill our matrix A
    a = 1/(eps_zx_mat*mu_yx_mat*omega**2*dx**2);
    b = zeros(size(a));
    b(2:end,:) = (1/(eps_zx_mat[1:-1,:]*
        mu_yx_mat(1:end-1,:)*omega**2*dx**2));
    
    c = 1/(eps_zy_mat*mu_xy_mat*omega**2*dx**2);
    d = zeros(size(c));
    d(:,2:end) = (1/(eps_zy_mat[:,1:-1]*
        mu_xy_mat[:,1:end-1]*omega**2*dx**2));
    
    e = 1-(a+b+c+d);
    #have to use 2 to end here for -1 values
    f = zeros(size(e));
    f[1:-1,1:-1] = ((eps0-eps_zi_mat[1:-1,1:-1])/
            (eps_zi_mat[1:-1,1:-1])*Ezi[1:-1,1:-1] + 
        (mu0-mu_yi_mat[1:-1,1:-1])/
            (1i*omega*dx*eps_zx_mat[1:-1,1:-1]
            *mu_yi_mat[1:-1,1:-1])*Hyi[1:-1,1:-1] - 
        (mu0-mu_yi_mat[0:-2,1:-1])/
            (1i*omega*dx*eps_zx_mat[1:-1,1:-1]
            *mu_yi_mat[0:-2,1:-1])*Hyi[0:-2,1:-1] - 
        (mu0-mu_xi_mat[1:-1,1:-1])/
            (1i*omega*dy*eps_zy_mat[1:-1,1:-1]
            *mu_xi_mat[1:-1,1:-1])*Hxi[1:-1,1:-1] + 
        (mu0-mu_xi_mat[1:-1,0:-2])/
            (1i*omega*dy*eps_zy_mat[1:-1,1:-1]
            *mu_xi_mat[1:-1,0:-2])*Hxi[1:-1,0:-2]);
>>>>>>> refs/remotes/origin/master
    
    #now we can clear all of our intermediate matrices
    #clear eps_zx_mat eps_zy_mat eps_zi_mat
    #clear mu_xy_mat mu_yx_mat mu_xi_mat mu_yi_mat
    
    # Setting up our final matrices including the sparse matrix
    #now lets build our final solvable matrices and vectors
    #now we will reshape all of our other coefficients
<<<<<<< HEAD
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
=======
    ar = reshape(a,[1,(cells_x)*(cells_y)]); #(i+1,j)
    br = reshape(b,[1,(cells_x)*(cells_y)]); #(i-1,j)
    cr = reshape(c,[1,(cells_x)*(cells_y)]); #(i,j+1)
    dr = reshape(d,[1,(cells_x)*(cells_y)]); #(i,j-1)
    er = reshape(e,[1,(cells_x)*(cells_y)]); #(i,j)
    
    #these will be y in Ax=y
    fr = reshape(f,[1,(cells_x)*(cells_y)]); #(i,j)
>>>>>>> refs/remotes/origin/master
    
    #clear a b c d e f
    
    #lets declare the size of 1 dimension of our sparse mat
    sparse_n = cells_x*cells_y; 
    
    #now lets create the indices for each of our values
    n    = np.arange(0,sparse_n);     #use this to modulo for wrapping around
    yIdx = n;          				  #indices in python range
<<<<<<< HEAD
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
    x = splinalg.spsolve(A,fr);
    
    #reshape back into a matrix
    E_scat = np.reshape(x,(cells_y,cells_x));
    
    #create our total field
    E_tot = E_scat.transpose()+np.conj(Ezi);
    return E_tot
    #rv = 0;
    

#%% Plotting
if __name__=='__main__':
    FDFD_2D()
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

xp = np.arange(cells_x)*dx;yp = np.arange(cells_y)*dy;
[Y,X] = np.meshgrid(xp,yp);

fig = make_subplots(rows=1,cols=3,
                    specs=[[{'type': 'surface'},{'type': 'surface'},{'type':'scatter'}]],
                    subplot_titles=("Incident Field","Total Field"))
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
fig.add_trace(go.Scatter(x=np.arange(len(fr)),y=fr.flatten().real),row=1,col=3)
fig.show(renderer='browser')
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

=======
    cIdx = mod(n+cells_x,sparse_n)+1; #(i,j+1)
    dIdx = mod(n-cells_x,sparse_n)+1; #(i,j-1)
    aIdx = mod(n-1,sparse_n)+1;       #(i+1,j)
    bIdx = mod(n+1,sparse_n)+1;       #(i-1,j)
    eIdx = n+1;                       #(i,j)
    
    #concatenate for sparse builder
    sparse_vals  = [ar  ,br  ,cr  ,dr  ,er  ];
    sparse_x_idx = [aIdx,bIdx,cIdx,dIdx,eIdx];
    sparse_y_idx = [yIdx,yIdx,yIdx,yIdx,yIdx];
    
    #now build the sparse matrix
    A = sparse(sparse_y_idx,sparse_x_idx,sparse_vals);
    
    #solve for our scatterfield
    x = A\fr';
    
    #reshape back into a matrix
    E_scat = reshape(x,[cells_x,cells_y]);
    
    #create our total field
    E_tot = E_scat+conj(Ezi);
    rv = 0;
>>>>>>> refs/remotes/origin/master
