function [bfr,azr,elr] = beamform_speed(num_angles,varargin)
%@brief This function is used to calculate beamforming for given number of
%   input angles per plane (az,el). Good for speed testing of languages.
%@note this is calculated for a 35 by 35 element equispaced planar array
%   and includes time required to both synthesize the data for a single
%   incident plane wave and calculate beamforming values at different
%   angles.
%@param[in] num_angles - number of angles to calculate on each plane
%   (az,el). The total number of angles will be num_angles^2 after meshgridding
%@param[in/OPT] dtype - data type to use (default @double)
%@return [beamformed values, azimuth angles, elevation angles]

%first add our libraries (this has to be done inside this function)
mypath = fileparts(mfilename('fullpath')); %get the directory of this file
addpath(fullfile(mypath,'../../aoa/MATLAB/serial'));
addpath(fullfile(mypath,'../../aoa/MATLAB/generic'));

%parse our input args
p = inputParser();
addOptional(p,'dtype',@double);
parse(p,varargin{:});
dtype = p.Results.dtype;

%now lets actually calculate
%freqs = dtype([40e9]); %frequency
%freqs = linspace(26.5,40e9,25);
freqs = np.arange(26.5e9,40e9,10e6)
numel = [35,35,1]; %number of elements in x,y

%now calculate everything
spacing = 2.99e8/max(freqs)/2; %get our lambda/2
[Yel,Xel,Zel] = meshgrid((0:numel(1)-1)*spacing,(0:numel(2)-1)*spacing,(numel(3)-1)*spacing); %create our positions
pos = dtype([reshape(Xel,[],1),reshape(Yel,[],1),reshape(Zel,[],1)]); %get our position [x,y,z] list

%create all of our angles
azi = dtype(deg2rad(linspace(-90,90,num_angles)));
eli = dtype(deg2rad(linspace(-90,90,num_angles)));
[AZ,EL] = meshgrid(azi,eli);

%flatten the arrays for beamforming
az = reshape(AZ,1,[]);
el = reshape(EL,1,[]);

%calcualte weights and synthesize data
weights = complex(dtype(ones(1,size(pos,1)))); %get our weights
sv = synthesize_data(freqs(1),pos,-pi/4,pi/4);
msv = dtype(sv);
meas_vals = repmat(msv,length(freqs),1);

%get our beamformed values
bf_vals = get_beamformed_values(freqs,pos,weights,meas_vals,az,el);

%reshape our values for returning
azr = reshape(az,[],length(azi),length(eli));
elr = reshape(el,[],length(azi),length(eli));
bfr = reshape(bf_vals,[],length(azi),length(eli));

end

%{
[bfv,az,el] = beamform_speed(181);
surf(az,el,10*log10(abs(bfv)));
shading('interp');
view([0,90])

tic; beamform_speed(181); toc

tic; beamform_speed(181,@single); toc

%}
