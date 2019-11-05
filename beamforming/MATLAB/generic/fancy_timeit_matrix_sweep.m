function [stats,ret_vals] = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,dim_range,num_reps,varargin)
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

    defaultTimerFunct = @fancy_timeit;
    defaultArgFunct = @arg_gen_funct;
    defaultDtype = @double;
    %parse the variable args
    p = inputParser;
    addParameter(p,'arg_gen_funct',defaultArgFunct);
    addParameter(p,'timer_funct',defaultTimerFunct);
    addParameter(p,'dtype',defaultDtype);
    parse(p,varargin{:});
    %now create our statistics structure
    stats = struct;
    ret_vals = struct; %return values (last value in each function)
    for i=1:length(funct_names)
        stats.(funct_names{i}) = struct;
    end
    max_num_args = max(num_arg_list); %maximum number of input arguments to any funct
    %now lets iterate through each size of matrix
    for d=1:length(dim_range)
        %first create our random matrices
        dim = dim_range(d); %get our dimension
        fprintf("Running with matrix of %dx%d:\n",dim,dim);
        %create our input arguments
        arg_inputs = p.Results.arg_gen_funct(dim,max_num_args);
        %now lets loop through each function
        for f=1:length(funct_list)
            funct = funct_list{f};
            funct_name = funct_names{f}; %get the function name as a string
            num_args = num_arg_list(f);
            %now create the string of the function to run with args
            run_str = ['@() ',func2str(funct),'(',strip(sprintf('arg_inputs{%d},',1:num_args),','),')'];
            %now time the run
            fprintf('    %10s :',funct_name);
            [cur_stat,rv] = p.Results.timer_funct(eval(run_str),num_reps);
            stats.(funct_name).(['m_',num2str(dim)]) = cur_stat;
            ret_vals.(funct_name) = rv;
            fprintf(' SUCCESS\n')
        end
    end

end

function [args] = arg_gen_funct(dim,num_args)
    args = {};
    for i=1:num_args
        args{end+1} = rand(dim)+1i*rand(dim);
    end
end