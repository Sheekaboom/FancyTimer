%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   In Class assignment Comp-EM Fall 2017       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
%-----------------------------------------------%
%% First lets define some values needed to build%
%  the grid      (changed by user)              %
%-----------------------------------------------%

%set the resolution of our grid in meters
dx = 1e-3;dy = 1e-3;
startx = -10e-2;endx = 10e-2;
starty = -10e-2;endy = 10e-2;
x = startx:dx:endx;
y = starty:dy:endy;

%our operation frequency
freq = 6e9;
omega = 2*pi*freq;
inc_amp = 1; %incident wave amplitude

%-----------------------------------------------%
%% Now lets build some problem space vairables  %
%-----------------------------------------------%
%calculate our wavelength and wavenumber
c      = 2.99792458e8;
lambda = c./freq;
kx     = (2*pi)./lambda;

%now location values
%this gives us our locational indexes
[X,Y] = meshgrid(x,y);
Xdim = X;
Ydim = Y;


%-----------------------------------------------%
%% Now lets set up and define our surce matrices%
%-----------------------------------------------%

Ezi = inc_amp.*exp(-1i*kx*Xdim);

%-----------------------------------------------%
%% And plot them                                %
%-----------------------------------------------%

% mag = abs(Ezi);
% phase = angle(Ezi).*180./(2*pi);
% phase = atan2(imag(Ezi),real(Ezi)).*180./(2*pi);
% 
% subplot(2,2,1);
% surf(real(Ezi));shading interp;
% title("Amplitude");
% subplot(2,2,2);
% surf(phase);shading interp;
% subplot(2,2,3);
% title("phase");
% plot(real(Ezi(1,:)));grid on;
% subplot(2,2,4);
% plot(phase(1,:));grid on;

%---------END SOURCE--------------------

%% Begin solving

%We already have our mesh grid for XY
%calculate each of our angles and dists
%in rect coords

phi = atan2(Y,X); %angle
rho = sqrt(X.^2 + Y.^2); %dist
Brho = rho;%kx*rho;
nn = 80;
%cannot get this to change the size of cylinder
%and still be stable
Ba = 2e-3;

n = -nn:nn;
n=reshape(n,[1,1,size(n,2)]); %reshape to get 3d from mult

%cn = -j^-n (Jn(Ba)/Hn(Ba))e^j*n*phi
cn = ((-1i).^-n).*(besselj(n,Ba)./besselh(n,2,Ba)).*exp(-1i.*n.*phi);
%compute our bessel values for each rho
for ni=1:size(n,3)
    Hrho(:,:,ni) = besselh(n(ni),2,Brho);
    disp(['i=' num2str(n(ni))]);
end
Ezs = Ezi.*sum(cn.*Hrho,3);
clear sumEzs
%Es = E0*sum(-inf,inf,cnHn^2(b*phi)))

%Cn = 0 in boundary
%n = 3+3Ba
%k1 = B1 = sqrt(epsr)*Bo (Bo = 1)
%?Ba = sqrt(eps)
%phi0 = 180
%what is hat(a)_z and E_0
%what is beta_rho?
