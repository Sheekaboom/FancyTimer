function [beamformed_values,az_angles,el_angles] = calculate_beamforming(num_angles)
%@brief This function is used to calculate beamforming for given number of
%   input angles per plane (az,el). Good for speed testing of languages.
%@note this is calculated for a 35 by 35 element equispaced planar array
%   and includes time required to both synthesize the data for a single
%   incident plane wave and calculate beamforming values at different
%   angles.
%@param[in] num_angles - number of angles to calculate on each plane
%   (az,el). The total number of angles will be num_angles^2 after meshgridding
%@return [beamformed values, azimuth angles, elevation angles]

%first add our libraries
mypath = fileparts(mfilename('fullpath')); %get the directory of this file
addpath(fullfile(mypath,'../../aoa/MATLAB/serial'));

%now lets actually calculate
addpath('../generic');
addpath('C:\Users\aweis\git\AoA_Estimation\base\data_synthesis\');
freqs = [40e9]; %frequency
%freqs = np.arange(26.5e9,40e9,10e6)
spacing = 2.99e8/max(freqs)/2; %get our lambda/2
%numel = [35,35,1]; %number of elements in x,y
numel = [10,10,1];
[Yel,Xel,Zel] = meshgrid((0:numel(1)-1)*spacing,(0:numel(1)-1)*spacing,(numel(3)-1)*spacing); %create our positions
pos = [reshape(Xel,[],1),reshape(Yel,[],1),reshape(Zel,[],1)]; %get our position [x,y,z] list
azi = deg2rad(-90:90-1);
eli = deg2rad(-90:90-1);
[AZ,EL] = meshgrid(azi,eli);
az = reshape(AZ,1,[]);
el = reshape(EL,1,[]);
%az = deg2rad([45,50]);
%el = deg2rad([0,0]);
weights = complex(ones(1,size(pos,1))); %get our weights
tic;
%sv = get_steering_vectors(freqs(1),pos,az,el);
sv = synthesize_data(freqs(1),pos,-pi/4,pi/4);
t = toc;
msv = (sv);
meas_vals = repmat(msv,length(freqs),1);

bf_vals = get_beamformed_values(freqs,pos,weights,meas_vals,az,el);

azr = reshape(az,length(azi),length(eli));
elr = reshape(el,length(azi),length(eli));
bfr = reshape(bf_vals,length(azi),length(eli));
beamformed_values = get_beamformed_values(freqs,positions,weights,meas_vals,az,el);

end

