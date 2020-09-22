%% some setup
if ~exist('pycom_path','var') % check for pycom path
    file_dir = fileparts(mfilename('fullpath')); %get the directory of this file
    pycom_path = fullfile(file_dir,'../../'); %assume the base is 2 levels up from this
end

if ~exist('DEBUG','var') % Check for debug flag
    DEBUG = false;
    fprintf("DEBUG Not defined. Setting to False");
end

if ~exist('output_directory','var') % check for provided output directory
    output_directory = './data';
end

addpath(fullfile(pycom_path,'python_matlab_comparison\beamforming')); % add OperationTimer
addpath(fullfile(pycom_path,'base'));

%% Some initialization
if DEBUG
    dim_list = floor(linspace(1,50,6));
    num_reps = 5;
else
    dim_list = floor(linspace(1,500,51));
    num_reps = 100;
end
out_file = fullfile(output_directory,'stats_mat_beamforming.mat');

%% Test the fancy_timeit_matrix_sweep capability
functs      = {@beamform_speed};
funct_names = {'beamforming'  };
num_args    = [2      ];

% now run 
stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct);
stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct_single);
	
%% now save out
m_single = stats_mat_single; % matlab cant do sparse single
m_double = stats_mat_double;
save(out_file,'m_single','m_double');
    
%% functions for fft arg generation                                     
function args = fdfd_arg_gen_funct(dim,num_args)
     %@brief generate default arguments
    args = {dim,@double};
end

function args = fdfd_arg_gen_funct_single(dim,num_args)
     %@brief generate default arguments
    args = {dim,@single};
end
    
    
    
    
	
	
	