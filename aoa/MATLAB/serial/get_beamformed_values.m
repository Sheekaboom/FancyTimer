function [out_vals] = get_beamformed_values(freqs,positions,weights,meas_vals,az,el)
%@brief get beamformed values using matlab
%@param[in] freqs - np array of frequencies to beaform at
%@param[in] positions - xyz positions to perform at
%@param[in] weights - tapering to apply to each element
%@param[in] meas_vals - measured values at each point
%@param[in] az - array of azimuthal angles to calculate
%@param[in] el - array of elevations
%return beamformed values
num_pos = size(positions,1);
%determine data type
if isa(meas_vals,'double')
    dtype = @double;
else
    dtype = @single;
end
%now calculate
out_vals = dtype(zeros(length(freqs),length(az))); %is matlab column major?
for fn=1:length(freqs) %manually loop through all frequncies
    sv = get_steering_vectors(freqs(fn),positions,az,el);
    out_vals(fn,:) = sum(weights.*meas_vals(fn,:).*sv,2)./num_pos;
end
end

function [steering_vecs] = get_steering_vectors(freq,positions,az,el)
%@brief get the steering vectors using a serial approach
%@param[in] frequency - frequency to calculate at
%@param[in] positions - positions of elements
%@param[in] az - azimuth angles (raidans)
%@param[in] el - elevation angles (radians)
kvecs = get_k_vector_azel(freq,az,el); %get k vectors
steering_vecs = exp(-1i.*(positions*kvecs)).';
end

