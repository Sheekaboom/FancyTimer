function [kvec] = get_k_vector_azel(freq,az,el)
%@brief get our k vectors (e.g. kv_x = k*sin(az)*cos(el))
%@note azel here are in radians
%@note this assumes boresight is az=0 el=0
%@param[in] freq - frequency to calcaulte for 
%@param[in] az - azimuth value to calculate for in radians
%@param[in] el - elevation value to calculate for in radians
%@return k-vector for the angle and frequency
k = get_k(freq,1,1); %get the wavenumber
kvec = k.*[sin(az).*cos(el);sin(el);cos(az).*cos(el)];
end

%% Test code
%{
kvec = get_k_vector_azel(40e9,[-45,20,0],[45,0,3]);
out=[-374.7356914097283038,  713.3447664223891707,231.3504328238943799;...
     765.3567036207713272,    0.0000000000000000, 342.1107031197503829;...
     -0.0000000000000000,  118.3062665560215549, -829.9483383078243151]';
assert(all(all(kvec==out)));
%}

