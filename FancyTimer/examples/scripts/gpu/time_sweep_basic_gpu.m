%% 
addpath('A:\git\pycom\base');

mat_dim_list = floor(linspace(1,10000,51));
%mat_dim_list = floor(linspace(1,500,6));
out_file = 'stats_mat_basic_gpu.mat';

combo_fun   = @(x,y) arrayfun(@(a,b) a+b*exp(a),x,y);
functs      = {@plus ,@minus,@times ,@rdivide,@exp  ,combo_fun};
funct_names = {'add' ,'sub' ,'mult' ,'div'   ,'exp' ,'combo'  };
num_args    = [2    ,2    ,2     ,2      ,1    ,2        ];
num_reps    = 100;

% now run 
gpu_double = @(x) gpuArray(double(x));
gpu_single = @(x) gpuArray(single(x));
stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',gpu_double);
stats_single = OperationTimer.fancy_timeit_matrix_sweep(...
	functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',gpu_single);

%matlab cant have two things named the same...
m_single = stats_single;
m_double = stats_double;

save(out_file,'m_single','m_double');



    