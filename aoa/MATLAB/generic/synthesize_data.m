function [synthetic_data] = synthesize_data(freq,pos,az,el,varargin)
% @brief create synthetic data for an array.
%   This will generate synthetic data assuming an incident plane wave
% @param[in] freq - frequency to create the data for in hz
% @param[in] pos  - position of elements in meters [x1,y1,z1;...;xn,yn,zn]
% @param[in] az   - azimuth of plane wave to generate in radians
% @param[in] el   - elevation of plane wave to generate in radians
% @param[in] magnitude - set the magnitude of the plane wave (in linear, default is 1)
% @param[in/OPT]  varargin - key/val pairs as follows:
%           snr - signal to noise ratio of the signal (defult 0 (no noise))
%               This is in dB compared to magnitude 10*log10
%           noise_funct - noise function to use to create noise (default
%                       randn)
% @return array of synthteic data corresponding to each [xn,yn,zn] row
%% setup our input parsing
p = inputParser; %startup the input parser for varargs and arg checking
valid_pos = @(pos) size(pos,2)==3; %check whether we have xyz or not
defaultMagnitude = 1;
defaultSNR = inf;
defaultNoiseFunct = @randn;
addRequired(p,'pos',valid_pos);
addParameter(p,'magnitude',defaultMagnitude);
addParameter(p,'snr',defaultSNR);
addParameter(p,'noise_funct',defaultNoiseFunct);
parse(p,pos,varargin{:});

%% Now lets create our synthetic data
kvecs = get_k_vector_azel(freq,az,el); %get k vectors
synthetic_data = p.Results.magnitude.*exp(-1i.*(pos*kvecs)).';

%% now add the noise
if p.Results.snr~=inf %if not infinite snr add it
    snr_lin = 10^(p.Results.snr/10); %10^(snr/10)
    noise_mag = p.Results.magnitude/snr_lin; %magnitude of our noise
    noise_vec = noise_mag.*(randn(size(synthetic_data))+1i.*randn(size(synthetic_data)));
    synthetic_data = synthetic_data+noise_vec; %add noise to synthetic data
end

end

%% Test code. Copy and paste to terminal to test
%{
%% add our paths to our functions
addpath('../../beamforming/MATLAB');
%% set our frequency and array positions
freqs = [40e9]; %frequency
%freqs = np.arange(26.5e9,40e9,10e6)
spacing = 2.99e8/max(freqs)/2; %get our lambda/2
%numel = [35,35,1]; %number of elements in x,y
numel = [10,10,1];
[Yel,Xel,Zel] = meshgrid((0:numel(1)-1)*spacing,(0:numel(1)-1)*spacing,(numel(3)-1)*spacing); %create our positions
pos = [reshape(Xel,[],1),reshape(Yel,[],1),reshape(Zel,[],1)]; %get our position [x,y,z] list

%% set our azimuths and elevations to calculate for
azi = deg2rad(-90:90-1);
eli = deg2rad(-90:90-1);
[AZ,EL] = meshgrid(azi,eli);
az = reshape(AZ,1,[]);
el = reshape(EL,1,[]);

%% set our weights and data
weights = complex(ones(1,size(pos,1))); %get our weights
sv = synthesize_data(freqs(1),pos,-pi/4,pi/8);
msv = (sv);
meas_vals = repmat(msv,length(freqs),1);

%% beamform
bf_vals = get_beamformed_values(freqs,pos,weights,meas_vals,az,el);

%% plot the data
azr = reshape(az,length(azi),length(eli));
elr = reshape(el,length(azi),length(eli));
bfr = reshape(bf_vals,length(azi),length(eli));
fig = figure();
surf(azr,elr,abs(bfr));shading('interp');
xlabel('az');ylabel('el');view([0,90]);
colorbar;
%}


