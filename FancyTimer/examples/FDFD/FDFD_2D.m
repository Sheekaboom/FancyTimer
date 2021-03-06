%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Finite Difference Frequency Domain Solver   %
%                                               %
% Written Fall 2017 By Alec Weiss               %
%               For Intro to Computatinal EM    %
%for a TE mode of propogation                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E_tot,E_scat] = FDFD_2D(varargin)
%-----------------------------------------------%
%% First lets define some values needed to build%
%  the grid      (changed by user)              %
%-----------------------------------------------%
p = inputParser();
addOptional(p,'num_cells_x',-1);
addOptional(p,'num_cells_y',-1);
addOptional(p,'use_gpu',false);
parse(p,varargin{:});
num_cells_x = p.Results.num_cells_x;
num_cells_y = p.Results.num_cells_y;
use_gpu = p.Results.use_gpu;

%set sizes of things in our domain (in meters)
cyl_sz_r = 10e-2; %radius of cylinder
cyl_sz_x = 15e-2; %size of cylinder in x direction
cyl_sz_y = 10e-2; %size of cylinder in y direction
air_buffer_sz = 15e-2;%size of surrounding air buffer
pml_thickness = 10e-2;% size of PML regions

%set the resolution of our grid in meters
dx = 5e-3;dy = 5e-3;
if num_cells_x > 0
    size_x = cyl_sz_x+2*air_buffer_sz+2*pml_thickness;
    dx = size_x/num_cells_x;
end
if num_cells_y > 0
    size_y = cyl_sz_y+2*air_buffer_sz+2*pml_thickness;
    dy = size_y/num_cells_y;
end

%our permitivities and permeabilities
%of our cylinder
eps_cyl = 1e10;
mu_cyl  = 1;
eps_0   = 8.85418782e-12;
mu_0    = 1.25663706e-6;
eps0    = eps_0;
mu0     = mu_0;

%our operation frequency
freq = 3e9;
omega = 2*pi*freq;
inc_amp = 1; %incident wave amplitude

%calculate our wavelength and wavenumber
c0      = 2.99792458e8;
lambda = c0/freq;
kx     = (2*pi)/lambda;

%-----------------------------------------------%
%% Now lets set up our problem space variables  %
%-----------------------------------------------%
%size in cells of each of our regions
%size of our cylinders in each dim
cyl_cells_x = ceil(cyl_sz_x/dx);
cyl_cells_y = ceil(cyl_sz_y/dy);
%size of our air buffer in cells
airbuf_cells_x = ceil(air_buffer_sz/dx);
airbuf_cells_y = ceil(air_buffer_sz/dy);
%now pml region size in cells
pml_cells_x = ceil(pml_thickness/dx);
pml_cells_y = ceil(pml_thickness/dy);

%total size of our grid total grid
%just add up all of our pieces
%*2 by pml and airbuf because one on each side
cells_x = (cyl_cells_x + 2*airbuf_cells_x + 2*pml_cells_x);
cells_y = (cyl_cells_y + 2*airbuf_cells_y + 2*pml_cells_y);

%----------------------------------------------------%
%% We can now begin building our cofficient matrices %
%----------------------------------------------------%
%these matrices will later be translated into a sparse
%matrix and then be cleared from memory
%this just ensures values are set correctly and allows
%for faster (than looping) assignment to sparse matrix
%-------
%lets first create matrices from which we create other 
%values THESE WILL BE CLEARED after usage
eps_mat = ones(cells_x,cells_y).*eps0;
mu_mat  = ones(cells_x,cells_y).*mu0;

%initialize these to zeroes so
%we can multiply to get values easily
%get our maximum conductivity
sig_max      = .3; %maximum conductivity
n            = 2;
sigma_ex_mat = zeros(cells_x,cells_y);
sigma_ey_mat = zeros(cells_x,cells_y);
sigma_mx_mat = zeros(cells_x,cells_y);
sigma_my_mat = zeros(cells_x,cells_y);

if use_gpu
    eps_mat = gpuArray(eps_mat);
    mu_mat  = gpuArray(mu_mat);
    sigma_ex_mat = gpuArray(sigma_ex_mat);
    sigma_ey_mat = gpuArray(sigma_ey_mat);
    sigma_mx_mat = gpuArray(sigma_mx_mat);
    sigma_my_mat = gpuArray(sigma_my_mat);
end

%now we fill our sigma matrices
%use a parabolic conductivity
hxe = (1:pml_cells_x)'   .*ones(pml_cells_x,cells_y); %end h value
hxs = (pml_cells_x:-1:1)'.*ones(pml_cells_x,cells_y); %start h value
hye = (1:pml_cells_y)    .*ones(cells_x,pml_cells_y);
hys = (pml_cells_y:-1:1) .*ones(cells_x,pml_cells_y);

sigma_ex_mat(1:pml_cells_x,:) = sig_max.*(hxs/pml_cells_x).^(n+1);
sigma_ex_mat(end-pml_cells_x+1:end,:) = (sig_max.*(hxe/pml_cells_x).^(n+1));
sigma_ey_mat(:,1:pml_cells_y) = (sig_max.*(hys/pml_cells_y).^(n+1));
sigma_ey_mat(:,end-pml_cells_y+1:end) = (sig_max.*(hye/pml_cells_y).^(n+1));

sigma_mx_mat(1:pml_cells_x,:) = (sig_max.*((hxs+.5)/pml_cells_x).^(n+1).*mu0./eps0);
sigma_mx_mat(end-pml_cells_x+1:end,:) = (sig_max.*((hxe+.5)/pml_cells_x).^(n+1).*mu0./eps0);
sigma_my_mat(:,1:pml_cells_y) = (sig_max.*((hys+.5)/pml_cells_y).^(n+1).*mu0./eps0);
sigma_my_mat(:,end-pml_cells_y+1:end) = (sig_max.*((hye+.5)/pml_cells_y).^(n+1).*mu0./eps0);
%Now we set our geometry values

%CIRCULAR
cells_r = ceil(cyl_sz_r/dx);
%circular middle
icx = ceil(cells_x/2); %the center of the grid
icy = ceil(cells_y/2);
%meshgrid
[CY,CX] = meshgrid((1:cells_y),(1:cells_x));
cyl_ind = sqrt((CX-icx).^2+(CY-icy).^2)<=cells_r;

%first set our inner cylinder values 
eps_mat(cyl_ind) = eps_cyl.*eps_mat(cyl_ind);
mu_mat(cyl_ind)  = mu_cyl.*mu_mat(cyl_ind);

%now we can adjust the PML values for epsilon
eps_zx_mat = eps_mat + (sigma_ex_mat/(1i*omega));
eps_zy_mat = eps_mat + (sigma_ey_mat/(1i*omega));
eps_zi_mat = eps_mat;
%now lets do the same thing for mu vals
mu_xy_mat = mu_mat + (sigma_my_mat/(1i*omega));
mu_yx_mat = mu_mat + (sigma_mx_mat/(1i*omega));
mu_xi_mat = mu_mat;
mu_yi_mat = mu_mat;

%clear sigma_ex_mat sigma_ey_mat
%clear sigma_mx_mat sigma_my_mat
%clear eps_mat mu_mat;
%CHECKPOINT: WORKS TO HERE

%----------------------------------------------------%
%% Build our sources                                 %
%----------------------------------------------------%
%source Ez = Ae^(-j*k_x*x)
%Hy and Hx solved from Ez
%mesh in x direction used for plane wave
eta_x = sqrt(mu_xi_mat./eps_zi_mat);
eta_y = sqrt(mu_yi_mat./eps_zi_mat);
x_mesh = (((1:cells_x).*dx)-(ceil(cells_x/2).*dx))'.*ones(cells_x,cells_y);
Ezi = inc_amp.*exp(-1i.*kx.*x_mesh);
Hxi = Ezi.*1./eta_x;
Hyi = Ezi.*1./eta_y;

%clear eta_x eta_y

%CHECKPOINT: WORKS TO HERE

%----------------------------------------------------%
%% Create our coefficient matrices                   %
%----------------------------------------------------%
%these will be reshaped to fill our matrix A
a = 1./(eps_zx_mat.*mu_yx_mat.*omega^2.*dx^2);
b = zeros(size(a));
b(2:end,:) = (1./(eps_zx_mat(2:end,:).*mu_yx_mat(1:end-1,:).*omega^2.*dx^2));

c = 1./(eps_zy_mat.*mu_xy_mat.*omega^2.*dx^2);
d = zeros(size(c));
d(:,2:end) = (1./(eps_zy_mat(:,2:end).*mu_xy_mat(:,1:end-1).*omega^2.*dx^2));

e = 1-(a+b+c+d);
%have to use 2 to end here for -1 values
f = zeros(size(e));
f(2:end,2:end) = ((eps0-eps_zi_mat(2:end,2:end))./...
        (eps_zi_mat(2:end,2:end)).*Ezi(2:end,2:end) + ...
    (mu0-mu_yi_mat(2:end,2:end))./...
        (1i.*omega.*dx.*eps_zx_mat(2:end,2:end)...
        .*mu_yi_mat(2:end,2:end)).*Hyi(2:end,2:end) - ...
    (mu0-mu_yi_mat(1:end-1,2:end))./...
        (1i.*omega.*dx.*eps_zx_mat(2:end,2:end)...
        .*mu_yi_mat(1:end-1,2:end)).*Hyi(1:end-1,2:end) - ...
    (mu0-mu_xi_mat(2:end,2:end))./...
        (1i.*omega.*dy.*eps_zy_mat(2:end,2:end)...
        .*mu_xi_mat(2:end,2:end)).*Hxi(2:end,2:end) + ...
    (mu0-mu_xi_mat(2:end,1:end-1))./...
        (1i.*omega.*dy.*eps_zy_mat(2:end,2:end)...
        .*mu_xi_mat(2:end,1:end-1)).*Hxi(2:end,1:end-1));

%now we can clear all of our intermediate matrices
%clear eps_zx_mat eps_zy_mat eps_zi_mat
%clear mu_xy_mat mu_yx_mat mu_xi_mat mu_yi_mat

%% Setting up our final matrices including the sparse matrix
%now lets build our final solvable matrices and vectors
%now we will reshape all of our other coefficients
ar = reshape(a,[1,(cells_x)*(cells_y)]); %(i+1,j)
br = reshape(b,[1,(cells_x)*(cells_y)]); %(i-1,j)
cr = reshape(c,[1,(cells_x)*(cells_y)]); %(i,j+1)
dr = reshape(d,[1,(cells_x)*(cells_y)]); %(i,j-1)
er = reshape(e,[1,(cells_x)*(cells_y)]); %(i,j)

%these will be y in Ax=y
fr = reshape(f,[1,(cells_x)*(cells_y)]); %(i,j)

%clear a b c d e f

%lets declare the size of 1 dimension of our sparse mat
sparse_n = cells_x*cells_y; 

%now lets create the indices for each of our values
n      = 0:sparse_n-1; %use this to modulo for wrapping around
yIdx = n+1;          %indices in matlab range
cIdx = mod(n+cells_x,sparse_n)+1; %(i,j+1)
dIdx = mod(n-cells_x,sparse_n)+1; %(i,j-1)
aIdx = mod(n-1,sparse_n)+1;       %(i+1,j)
bIdx = mod(n+1,sparse_n)+1;       %(i-1,j)
eIdx = n+1;                       %(i,j)

%concatenate for sparse builder
sparse_vals  = [ar  ,br  ,cr  ,dr  ,er  ];
sparse_x_idx = [aIdx,bIdx,cIdx,dIdx,eIdx];
sparse_y_idx = [yIdx,yIdx,yIdx,yIdx,yIdx];

%now build the sparse matrix
A = sparse(sparse_y_idx,sparse_x_idx,sparse_vals);

%solve for our scatterfield
x1 = A\fr';

%reshape back into a matrix
E_scat = reshape(x1,[cells_x,cells_y]);

%create our total field
E_tot = E_scat+conj(Ezi);
end


%% calculate and  plot
%{

%calculate
num_cells = 130;
[E_tot,E_scat] = FDFD_2D(num_cells,num_cells);

%then plot
subplot(2,1,1);
surf(abs(E_scat)');
title('Scattered Field');
view([0,90]);
shading interp;
subplot(2,1,2);
surf(abs(E_tot)');
title('Total Field Magnitude');
view([0,90]);
shading interp;

fig = figure(); shading interp;
surf(abs(E_tot)');
title('Total Field Magnitude');
view([0,90]);

%}









