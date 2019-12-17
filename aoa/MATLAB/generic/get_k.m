function [k] = get_k(freq,eps_r,mu_r)
%@brief get our wavenumber
%@param[in] freq - frequency to calculate at
%@param[in] eps_r - relative permittivity
%@param[in] mu_r - relative permeability
lam = physconst('LightSpeed')/sqrt(eps_r*mu_r)/freq;
k = 2*pi/lam;
end

%% Test code
%{
k=get_k(40e9,1,1);
assert(k==838.3380087806727)
%}