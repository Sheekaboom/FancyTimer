function [time_struct,rv] = fancy_timeit(funct_to_time,num_reps)
    % @brief return a list of times from num_reps
    % @param[in] funct_to_time - lambda function to call
    % @param[in] num_reps -how many times to repeat
    % @return list of times from each repeat measurement and the rv from the
    %   last run
    time_list = zeros(1,num_reps);
    for r=1:num_reps
        tic;
        rv = funct_to_time();
        time_list(r) = toc;
    end
    %% now calulate the statistics and pack in a structure
    time_struct.mean = mean(time_list);
    time_struct.stdev = std(time_list);
    time_struct.count = length(time_list);
    time_struct.min = min(time_list);
    time_struct.max = max(time_list);
    time_struct.range = max(time_list)-min(time_list);
end

