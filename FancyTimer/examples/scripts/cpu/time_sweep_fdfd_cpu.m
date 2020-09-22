path_to_pycom = 'A:\git\';
addpath(fullfile(path_to_pycom,'pycom\python_matlab_comparison\FDFD\'));
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

addpath(fullfile(pycom_path,'base')); % add OperationTimer
addpath(fullfile(pycom_path,'python_matlab_comparison\FDFD\'));

%% Some initialization
if DEBUG
    dim_list = floor(linspace(1,50,6));
    num_reps = 5;
else
    dim_list = floor(linspace(1,500,51));
    num_reps = 100;
end
out_file = fullfile(output_directory,'stats_mat_fdfd.mat');

%% Test the fancy_timeit_matrix_sweep capability
functs      = {@FDFD_2D};
funct_names = {'fdfd'  };
num_args    = [2      ];

% now run 
stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct);
	
%% now save out
m_single = struct(); % matlab cant do sparse single
m_double = stats_mat_double;
save(out_file,'m_single','m_double');
    
%% functions for fft arg generation                                     
function args = fdfd_arg_gen_funct(dim,num_args)
     %@brief generate default arguments
    args = {dim,dim};
end

function new_struct = update_struct(struct_to_update,struct_to_add)
    %@brief Add all of the fields of struct_to_add to struct_to_update
    %@note If overlapping fields exist, struct_to_update will be overwritten
    %@param[in] struct_to_update - what struct are we updating
    %@param[in] struct_to_add - struct with fields to add to struct_to_update
    %@return New structure with fields from both input structs
    sta_fldnames = fieldnames(struct_to_add);
    new_struct = struct_to_update; %copy struct to update
    for i=1:length(sta_fldnames) %add all of our fieldnames
        fldname = sta_fldnames{i};
        new_struct.(fldname) = struct_to_add.(fldname); %add to new struct
    end
end
    
    
    
    
	
	
	