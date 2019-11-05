
%mat_dim_list = 2.^(4:14);
mat_dim_list = floor(logspace(1,4,100));

%% Test the fancy_timeit_matrix_sweep capability
funct_list   = {@madd,@msub,@times,@rdivide};
funct_names  = {'add','sub','mult','div'   };
num_arg_list = [2    ,2    ,2     ,2       ];
[mat_stats_1,mat_rv_1] = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,100);

funct_list   = {@lu ,@mmult  };
funct_names  = {'lu','matmul'};
num_arg_list = [1   ,2       ];
[mat_stats_2,mat_rv_2] = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,10);


%% Now test on GPU calls
funct_list   = {@madd,@msub,@times,@rdivide};
funct_names  = {'add','sub','mult','div'   };
num_arg_list = [2    ,2    ,2     ,2       ];
[mat_gpu_stats_1,mat_gpu_rv_1] = fancy_timeit_gpu_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,100);


funct_list   = {@mmult  };
funct_names  = {'matmul'};
num_arg_list = [2       ];
[mat_gpu_stats_2,mat_gpu_rv_2] = fancy_timeit_gpu_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,10);

%% Now save out the data
save('mat_stats.mat','mat_stats_1','mat_stats_2','mat_gpu_stats_1','mat_gpu_stats_2');