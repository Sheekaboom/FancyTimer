function [time_struct,rv] = fancy_timeit_gpu(funct_to_time,num_reps)
% @brief return a list of times from num_reps
% @param[in] funct_to_time - lambda function to call
% @param[in] num_reps -how many times to repeat
time_list = zeros(1,num_reps);
for r=1:num_reps
    time_list(r) = gputimeit(funct_to_time); 
    rv=0;
end
%% now calulate the statistics and pack in a structure
time_struct.mean = mean(time_list);
time_struct.stdev = std(time_list);
time_struct.count = length(time_list);
time_struct.min = min(time_list);
time_struct.max = max(time_list);
time_struct.range = max(time_list)-min(time_list);
end

