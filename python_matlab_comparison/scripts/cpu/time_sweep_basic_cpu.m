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

%% now lets setup and run our test

if DEBUG
    dim_list = floor(linspace(1,500,6));
    num_reps = 10;
else
    dim_list = floor(linspace(1,10000,51));
    num_reps = 100;
end
out_file = fullfile(output_directory,'stats_mat_basic.mat');

%% now run
mat_dim_list = dim_list;

combo_fun   = @(x,y) x+y.*exp(x);
functs      = {@plus ,@minus,@times ,@rdivide,@exp  ,combo_fun};
funct_names = {'add' ,'sub' ,'mult' ,'div'   ,'exp' ,'combo'  };
num_args    = [2    ,2    ,2     ,2      ,1    ,2        ];

% now run 
stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
stats_single = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);

%matlab cant have two things named the same...
m_single = stats_single;
m_double = stats_double;

save(out_file,'m_single','m_double');



    