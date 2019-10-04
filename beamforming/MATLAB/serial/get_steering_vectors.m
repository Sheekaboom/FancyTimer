function [steering_vecs] = get_steering_vectors(freq,positions,az,el)
%@brief get the steering vectors using a serial approach
%@param[in] frequency - frequency to calculate at
%@param[in] positions - positions of elements
%@param[in] az - azimuth angles (raidans)
%@param[in] el - elevation angles (radians)
kvecs = get_k_vector_azel(freq,az,el); %get k vectors
steering_vecs = exp(-1i.*(positions*kvecs)).';
end

