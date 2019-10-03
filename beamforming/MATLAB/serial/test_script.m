addpath('../generic');
freqs = [40e9]; %frequency
%freqs = np.arange(26.5e9,40e9,10e6)
spacing = 2.99e8/max(freqs)/2; %get our lambda/2
%numel = [35,35,1]; %number of elements in x,y
numel = [35,35,1];
[Yel,Xel,Zel] = meshgrid((0:numel(1)-1)*spacing,(0:numel(1)-1)*spacing,(numel(3)-1)*spacing); %create our positions
pos = [reshape(Xel,[],1),reshape(Yel,[],1),reshape(Zel,[],1)]; %get our position [x,y,z] list
%az = (-90:90-1);
%el = (-90:90-1);
az = deg2rad([45,50]);
el = deg2rad([0,0]);
weights = complex(ones(1,size(pos,1))); %get our weights
sv = get_steering_vectors(freqs(1),pos,az,el);
msv = mean(sv);
meas_vals = repmat(msv,length(freqs),1);
bf_vals = get_beamformed_values(freqs,pos,weights,meas_vals,az,el);
