%This is a class for timing of functions with provided inputs
classdef OperationTimer < handle
    
    properties
        prop
    end
    
    methods(Static)
        
        function timer_stats = fancy_timeit(mycallable,num_reps,varargin)
            %@brief easy timeit function that will return the results of a function
            %    along with a dictionary of timing statistics
            %@param[in] mycallable - callable statement to time
            %@param[in/OPT] num_reps - number of repeats for timing and statistics
            p = inputParser();
            defaultNumReps = 3;
            addRequired(p,'mycallable', @(x) isa(x,'function_handle'));
            addOptional(p,'num_reps',defaultNumReps,@(x) x>0);
            parse(p,mycallable,num_reps,varargin{:});
            times = ones(1,p.Results.num_reps)*-3.14;
            for n = 1:p.Results.num_reps
                tic; %start timer
                p.Results.mycallable(); %run the function
                times(n) = toc; %gather the times
            end
            timer_stats = OperationTimer.calculate_statistics(times); %calculate timer stats
        end
        
        function stats_struct = calculate_statistics(time_list)
            %@brief calculate and store time stats
            stats_struct = struct();
            stats_struct.mean  = mean(time_list);
            stats_struct.stdev = std(time_list);
            stats_struct.count = length(time_list);
            stats_struct.min   = min(time_list);
            stats_struct.max   = max(time_list);
            stats_struct.range = range(time_list);
            stats_struct.raw   = time_list;
        end
            

        function stats = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,dim_range,num_reps,varargin)
            %@brief easy timeit function that will return the results of a function
            %    along with a dictionary of timing statistics
            %@param[in] funct_list - list of function handles to run (cellarray)
            %@param[in] funct_names - list of names for storing results
            %        Explicitly providing these provides uniformity between  libs (cellarray)
            %@param[in] num_arg_list - list of the number of input arguments (matrices)
            %        that should be passed to the corresponding function in myfunct_list(array)
            %@param[in] dim_range - range of numbers to use as M for an MxM matrix
            %@param[in/OPT] num_reps - number of repeats for timing and statistics
            %@param[in/OPT] kwargs - keyword arguments as follows
            %    arg_gen_funct -function to generate the matrix from a dim value input
            %        if not included default to generate np.cdouble random matrix
            %    timer_funct - function to use for timing. If not included uses fancy_timeit
            %    dtype - dtype to use for the default arg_gen_funct. should
            %       be @double or @single
            %    cleanup_funct = function to run on arg_inputs after each dimension iteration
            %        must recieve list of args to cleanup
            p = inputParser();
            defaultNumReps = 3;
            defaultArgGenFunct = @OperationTimer.default_arg_gen_funct;
            addRequired(p,'funct_list'  , @(x) all(cellfun(@(y) isa(y,'function_handle'),x)));
            addRequired(p,'funct_names' );
            addRequired(p,'num_arg_list');
            addRequired(p,'dim_range'   );
            addOptional(p,'num_reps'    ,defaultNumReps,@(x) x>0);
            addParameter(p,'arg_gen_funct',defaultArgGenFunct);
            addParameter(p,'timer_funct',@OperationTimer.fancy_timeit);
            addParameter(p,'cleanup_funct',@(x) 1);
            addParameter(p,'dtype',@double);
            parse(p,funct_list,funct_names,num_arg_list,dim_range,num_reps,varargin{:});
            
            %ret_vals = struct();
            stats = struct();
            max_num_args = max(num_arg_list); %get the maximum number of arguments
            %now loop through each size of our matrix
            for dn = 1:length(dim_range) %loop through each of the dimensions specified
                dim = dim_range(dn);
                %first create our random matrices
                if nargin(p.Results.arg_gen_funct)==3
                    arg_inputs = p.Results.arg_gen_funct(dim,max_num_args,p.Results.dtype);
                else
                    arg_inputs = p.Results.arg_gen_funct(dim,max_num_args);
                end
                if all(size(arg_inputs{1})==1) && ~issparse(arg_inputs{1})
                    fprintf(['Running with input of :',sprintf('%d,',arg_inputs{1}(1))]);
                else
                    fprintf(['Running with matrix of :',sprintf('%d,',size(arg_inputs{1}))]);
                end
                fprintf(' dtype= %s\n',class(arg_inputs{1}));
                %now run each function
                for fn=1:length(funct_list)
                    funct = funct_list{fn};
                    funct_name = funct_names{fn};
                    num_args = num_arg_list(fn);
                    fprintf('    %10s :',funct_name);
                    lam_fun = @() funct(arg_inputs{1:num_args});
                    cur_stat = p.Results.timer_funct(lam_fun,p.Results.num_reps);
                    if ~isfield(stats,funct_name)
                        stats.(funct_name) = struct();
                    end
                    stats.(funct_name).(['m_',num2str(dim)]) = cur_stat;
                    fprintf(' SUCCESS\n');
                p.Results.cleanup_funct(arg_inputs); %pass in list of args
                end
            end
        end
            
        function args = default_arg_gen_funct(dim,num_args,dtype_fun)
            %@brief generate default arguments
            args = {};
            for argn=1:num_args
                args{end+1} = dtype_fun(rand(dim,dim)+1i*rand(dim,dim));
            end
        end
        
    end %methods (Static)
    
end %class
           
      


        
        




