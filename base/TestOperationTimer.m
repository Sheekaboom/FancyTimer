%% some testing  
%myot = OperationTimer();
%% Test the fancy_timeit_matrix_sweep capability
mat_dim_list = 2.^(4:8);
%mat_dim_list = np.floor(np.logspace(1,4,100)).astype(np.int32)
funct_list   = {@plus,@minus,@times,@rdivide};
funct_names  = {'add','sub' ,'mult','div'   };
num_arg_list = [2     ,2    ,2     ,2       ];
stats = OperationTimer.fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,100);

% funct_list   = {@lu ,@mtimes  };
% funct_names  = {'lu','matmul' };
% num_arg_list = [1   ,2        ];
% [py_stats_2,rv] = OperationTimer.fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,10);
