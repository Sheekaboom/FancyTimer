%% some setup
if ~exist('pycom_path','var') % check for pycom path
    file_dir = fileparts(mlfilename('fullpath')); %get the directory of this file
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
    num_reps = 5;
else
    dim_list = floor(linspace(1,10000,51));
    num_reps = 25;
end

out_file = fullfile(output_directory,'stats_mat_matrix.mat');

%% Test the fancy_timeit_matrix_sweep capability
%lu_factor is used as opposed to lu to match the output of MATLAB's lu() function
functs      = {@lu   ,@mtimes };
funct_names = {'lu'  ,'matmul'};
num_args    = [1     ,2       ];
mat_dim_list = dim_list;

% now run 
stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);

%% and now matrix solving 
if DEBUG
    dim_list = floor(linspace(1,500,6));
    num_reps = 5;
else
    dim_list = floor(linspace(1,5000,51));
    num_reps = 25;
end

solve_reps = num_reps;
solve_dim_list = dim_list;
			 
solve_stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
        {@mldivide},{'solve'},[2],solve_dim_list,solve_reps...
        ,'arg_gen_funct',@solve_arg_gen_funct);
solve_stats_single = OperationTimer.fancy_timeit_matrix_sweep(...
        {@mldivide},{'solve'},[2],solve_dim_list,solve_reps...
        ,'arg_gen_funct',@solve_arg_gen_funct_single);
		
stats_mat_double = update_struct(stats_mat_double,solve_stats_double);
stats_mat_single = update_struct(stats_mat_single,solve_stats_single);

%% now save out in case sparse does something stupid
m_single = stats_mat_single;
m_double = stats_mat_double;
save(out_file,'m_single','m_double');
clear single double
    
%% now save out
m_single = stats_mat_single;
m_double = stats_mat_double;
save(out_file,'m_single','m_double');

%% solve data functions
%argument generation functions
function args = solve_arg_gen_funct(dim,num_args) %generate A and b for dense solving
    args = {};
    args{1} = double(rand(dim,dim)+1i*rand(dim,dim));
    args{2} = double(rand(dim,1)+1i*rand(dim,1));
 end
function args = solve_arg_gen_funct_single(dim,num_args) %generate A and b for dense solving
    args = {};
    args{1} = single(rand(dim,dim)+1i*rand(dim,dim));
    args{2} = single(rand(dim,1)+1i*rand(dim,1));
 end
         
%% funciton for updating a structure
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
    
    