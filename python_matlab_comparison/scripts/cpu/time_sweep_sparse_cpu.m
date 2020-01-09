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
    dim_list = floor(linspace(1,5000,51));
    num_reps = 25;
end

out_file = fullfile(output_directory,'stats_mat_sparse.mat');

%% and sparse solving. MATLAB cant do sparse single
ssolve_dim_list = dim_list;
ssolve_reps = num_reps;

%now get the results
ssolve_stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
        {@mldivide},{'ssolve'},[2],ssolve_dim_list,ssolve_reps...
        ,'arg_gen_funct',@ssolve_arg_gen_funct);
		
stats_mat_double = ssolve_stats_double;
ssolve_stats_double.notes = "THIS IS THE SAME DATA AS FOR DOUBLE. MATLAB DOES NOT SUPPORT SINGLE PRECICION SPARSE";
stats_mat_single = ssolve_stats_double; %simply save double as single
    
%% now save out
m_single = stats_mat_single;
m_double = stats_mat_double;
save(out_file,'m_single','m_double');
         
%% ssolve data functions
function spmat = generate_random_sparse(shape,num_el,dtype)
    %@brief fill a sparse array with numel_random elements
    data = rand(1,num_el)+1i*rand(1,num_el);
    ri = randi([1,shape(1)],1,num_el);
    ci = randi([1,shape(2)],1,num_el);
    spmat = dtype(sparse(ri,ci,data,shape(1),shape(2)));
end

function args = ssolve_arg_gen_funct(dim,num_args) %generate A and b for dense solving
    numel = floor(dim*5);
    args = {};
    args{1} = generate_random_sparse([dim,dim],numel,@double);
    args{2} = rand(dim,1)+1i*rand(dim,1);
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
    
    