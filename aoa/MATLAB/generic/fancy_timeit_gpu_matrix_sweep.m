function [stats,ret_vals] = fancy_timeit_gpu_matrix_sweep(funct_list,funct_names,num_arg_list,dim_range,num_reps,varargin)
    % @brief easy timeit function that will return the results of a function
    %     along with a dictionary of timing statistics
    % @param[in] funct_list - cellarray of function handles to run
    % @param[in] funct_names - cellarray of names for storing results
    %           Explicitly providing these provides uniformity between libs
    % @param[in] num_arg_list - array of the number of input arguments (matrices)
    %         that should be passed to the corresponding function in myfunct_list
    % @param[in] dim_range - range of numbers to use as M for an MxM matrix
    % @param[in/OPT] num_reps - number of repeats for timing and statistics
    % @param[in/OPT] kwargs - key value pairs as follows
    %     arg_gen_funct -function to generate the matrix from a dim value input
    %         if not included default to generate np.cdouble random matrix.
    %         Should return a cell array
    %     timer_funct - function to use for timing. If not included uses fancy_timeit
    %     dtype - dtype to use for the default arg_gen_funct. should be
    %               @double or @single
[stats,ret_vals] = fancy_timeit_matrix_sweep(...
    funct_list,funct_names,num_arg_list,dim_range,num_reps,...
    'timer_funct',@fancy_timeit_gpu,'arg_gen_funct',@arg_gen_funct_gpu);
end


function [args] = arg_gen_funct_gpu(dim,num_args)
    args = {};
    for i=1:num_args
        args{end+1} = gpuArray(rand(dim)+1i*rand(dim));
    end
end
