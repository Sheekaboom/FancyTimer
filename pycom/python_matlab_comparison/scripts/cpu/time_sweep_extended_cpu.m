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

%% Some initialization
if DEBUG
    dim_list = floor(linspace(1,500,6));
    num_reps = 10;
else
    dim_list = floor(linspace(1,10000,51));
    num_reps = 100;
end
out_file = fullfile(output_directory,'stats_mat_extended.mat');

%% Test the fancy_timeit_matrix_sweep capability
sum_fun = @(x) sum(sum(x));
functs      = {sum_fun};
funct_names = {'sum'  };
num_args    = [1      ];
mat_dim_list = dim_list;

% now run 
stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);
	
	
%% now measure our fft
if DEBUG
    dim_list = floor(linspace(1,50000,6));
    num_reps = 10;
else
    dim_list = floor(linspace(1,5000000,51));
    num_reps = 100;
end

fft_reps = num_reps;
fft_dim_list = dim_list;
    
stats_fft_double = OperationTimer.fancy_timeit_matrix_sweep(...
        {@fft},{'fft'},[1],fft_dim_list,fft_reps...
        ,'arg_gen_funct',@fft_arg_gen_funct);
stats_mat_double = update_struct(stats_mat_double,stats_fft_double);

stats_fft_single = OperationTimer.fancy_timeit_matrix_sweep(...
        {@fft},{'fft'},[1],fft_dim_list,fft_reps...
        ,'arg_gen_funct',@fft_arg_gen_funct_single);
stats_mat_single = update_struct(stats_mat_single,stats_fft_single);
    
%% now save out
m_single = stats_mat_single;
m_double = stats_mat_double;
save(out_file,'m_single','m_double');
    
%% functions for fft arg generation                                     
function args = fft_arg_gen_funct(dim,num_args)
     %@brief generate default arguments
    args = {};
    for argn=1:num_args
        args{end+1} = double(rand(1,dim)+1i*rand(1,dim));
    end
end

function args = fft_arg_gen_funct_single(dim,num_args)
     %@brief generate default arguments
    args = {};
    for argn=1:num_args
        args{end+1} = single(rand(1,dim)+1i*rand(1,dim));
    end
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
    
    
    
    
	
	
	