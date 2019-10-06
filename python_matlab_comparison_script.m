%% Compare matlab and python for the following
% Add/Sub/Mult/Div
% Exp, matmul, sum
% LU Decomp, fft, ODE?, Sparse?
m = 5000; n = 5000;
num_reps = 1;
a = rand(m,n)+1i*rand(m,n);
b = rand(m,n)+1i*rand(m,n);
a = real(a);
b = real(b);
%a = single(a);
%b = single(b);

%% Add/Sub/Mult/Div
%fprintf("Add/Sub/Mult/Div\n");
add_fun = @() a +b;
[add_double_complex_time,add_double_complex_rv] = fancy_timer(add_fun,num_reps);
display_time_stats(add_double_complex_time,'Add Complex Double')
% sub_fun = @() a -b;
% [sub_double_complex_time,sub_double_complex_rv] = fancy_timer(sub_fun,num_reps);
% mul_fun = @() a.*b;
% [mul_double_complex_time,mul_double_complex_rv] = fancy_timer(mul_fun,num_reps);
% div_fun = @() a./b;
% [div_double_complex_time,div_double_complex_rv] = fancy_timer(div_fun,num_reps);

%% exp/matmul/sum
%fprintf("exp/matmul/sum\n");
matmul_fun = @() a*b;
[matmul_double_complex_time,matmul_double_complex_rv] = fancy_timer(matmul_fun,num_reps);
display_time_stats(matmul_double_complex_time,'Matrix Multiply Complex Double')







%% LU Decomp
lu_fun = @() lu(a);
%fprintf("LU Decomposition\n");
[lu_double_complex_time,lu_double_complex_rv] = fancy_timer(lu_fun,num_reps);
display_time_stats(lu_double_complex_time,'LU Complex Double')



%% Timing functions
function [time_list,rv] = fancy_timer(funct_to_time,num_reps)
% @brief return a list of times from num_reps
% @param[in] funct_to_time - lambda function to call
% @param[in] num_reps -how many times to repeat
time_list = zeros(1,num_reps);
for r=1:num_reps
    tic;
    rv = funct_to_time();
    time_list(r) = toc;
end
end

function [] = display_time_stats(time_list,name)
% @brief print out statistics on our times 
% @param[in] time_list - list of recorded times
% @param[in] name - name to print info of
fprintf("%s : \n",name)
fprintf("    mean : %f\n",mean(time_list));
fprintf("    stdev : %f\n",std(time_list));
fprintf("    count : %f\n",length(time_list));
fprintf("    min : %f\n",min(time_list));
fprintf("    max : %f\n",max(time_list));
fprintf("    range : %f\n",max(time_list)-min(time_list));
end







