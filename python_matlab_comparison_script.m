%% Compare matlab and python for the following
% Add/Sub/Mult/Div
% Exp, matmul, sum
% LU Decomp, fft, ODE?, Sparse?
%out_file_path = './matlab_complex_double_times.json';
out_file_path = './matlab_complex_single_times.json';
m = 5000; n = 5000;
num_reps = 100;
rng(1234); %set the seed
a = rand(m,n)+1i*rand(m,n);
b = rand(m,n)+1i*rand(m,n);
%a = real(a);
%b = real(b);
a = single(a);
b = single(b);

%fft data
rng(1234)
fft_n = 2^23; %number of points in our fft (same as matlab bench
fft_data = rand(1,fft_n)+1i*rand(1,fft_n);
fft_data = single(fft_data);

ts_double = struct();

%% Add/Sub/Mult/Div
%fprintf("Add/Sub/Mult/Div\n");
add_fun = @() a +b;
[add_double_complex_time,add_double_complex_rv] = fancy_timer(add_fun,num_reps);
ts_double.add = display_time_stats(add_double_complex_time,'Add Complex Double');
sub_fun = @() a -b;
[sub_double_complex_time,sub_double_complex_rv] = fancy_timer(sub_fun,num_reps);
ts_double.sub = display_time_stats(sub_double_complex_time,'Sub Complex Double');
mul_fun = @() a.*b;
[mul_double_complex_time,mul_double_complex_rv] = fancy_timer(mul_fun,num_reps);
ts_double.mul = display_time_stats(mul_double_complex_time,'Mul Complex Double');
div_fun = @() a./b;
[div_double_complex_time,div_double_complex_rv] = fancy_timer(div_fun,num_reps);
ts_double.div = display_time_stats(div_double_complex_time,'Div Complex Double');

%% exp/matmul/sum
%fprintf("exp/matmul/sum\n");
matmul_fun = @() a*b;
[matmul_double_complex_time,matmul_double_complex_rv] = fancy_timer(matmul_fun,num_reps);
ts_double.matmul = display_time_stats(matmul_double_complex_time,'Matrix Multiply Complex Double');

exp_fun = @() exp(a);
[exp_double_complex_time,exp_double_complex_rv] = fancy_timer(exp_fun,num_reps);
ts_double.exp = display_time_stats(exp_double_complex_time,'Exponential Complex Double');

sum_fun = @() sum(a);
[sum_double_complex_time,sum_double_complex_rv] = fancy_timer(sum_fun,num_reps);
ts_double.sum = display_time_stats(sum_double_complex_time,'Sum Complex Double');

%% LU Decomp
lu_fun = @() lu(a);
%fprintf("LU Decomposition\n");
[lu_double_complex_time,lu_double_complex_rv] = fancy_timer(lu_fun,num_reps);
ts_double.lu = display_time_stats(lu_double_complex_time,'LU Complex Double');

%% FFT
fft_fun = @() fft(fft_data);
[fft_double_complex_time,fft_double_complex_rv] = fancy_timer(fft_fun,num_reps);
ts_double.fft = display_time_stats(fft_double_complex_time,'FFT Complex Double');

%% Combined operations
comb_fun = @() a+b.*exp(a);
[comb_double_complex_time,comb_double_complex_rv] = fancy_timer(comb_fun,num_reps);
ts_double.fft = display_time_stats(comb_double_complex_time,'Combined Complex Double');

%% Sparse Matrices
sm = 20000; sn = 20000;
num_sp_el = 1e6;
rng(1234)

sa = generate_random_sparse(sm,sn,num_sp_el);
sb = generate_random_sparse(sm,sn,num_sp_el);

sp_fun = @() sa*sb;
[sp_double_complex_time,sp_double_complex_rv] = fancy_timer(sp_fun,5);
ts_double.sparse = display_time_stats(sp_double_complex_time,'Combined Complex Double');

%% Beamforming
bf_wdir = './beamforming/MATLAB';
addpath(fullfile(bf_wdir,'./generic'));
addpath(fullfile(bf_wdir,'./serial'))
%freqs = [40e9]; %frequency
freqs = 26.5e9:100e6:39.9e9;
spacing = 2.99e8/max(freqs)/2; %get our lambda/2
%numel = [35,35,1]; %number of elements in x,y
numel = [10,10,1];
[Yel,Xel,Zel] = meshgrid((0:numel(1)-1)*spacing,(0:numel(1)-1)*spacing,(numel(3)-1)*spacing); %create our positions
pos = [reshape(Xel,[],1),reshape(Yel,[],1),reshape(Zel,[],1)]; %get our position [x,y,z] list
num_deg = 1000;
az = rand(1,num_deg);
el = rand(1,num_deg);
%az = deg2rad(-90:90-1);
%el = deg2rad(-90:90-1);
%[AZ,EL] = meshgrid(az,el);
%az = reshape(AZ,1,[]);
%el = reshape(EL,1,[]);
%az = deg2rad([45,50]);
%el = deg2rad([0,0]);
weights = complex(ones(1,size(pos,1))); %get our weights

sv = get_steering_vectors(freqs(1),pos,az,el);
msv = mean(sv);
meas_vals = repmat(msv,length(freqs),1);

freqs = single(freqs);
pos = single(pos);
weights = single(weights);
meas_vals = single(meas_vals);
az = single(az);
el = single(el);

bf_fun = @() get_beamformed_values(freqs,pos,weights,meas_vals,az,el);
[bf_double_complex_time,bf_double_complex_rv] = fancy_timer(bf_fun,10);
ts_double.beamform = display_time_stats(bf_double_complex_time,'Beamformed Complex Double');

%% FDFD
addpath(('./beamforming/python_matlab_comparison/FDFD'));
fdfd_fun = @() FDFD_2D();
[fdfd_double_complex_time,fdfd_double_complex_rv] = fancy_timer(fdfd_fun,10);
ts_double.fdfd = display_time_stats(fdfd_double_complex_time,'FDFD Complex Double');

%% Write all of the data out to a json file
write_json(ts_double,out_file_path);

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

function [time_struct] = display_time_stats(time_list,name)
% @brief print out statistics on our times 
% @param[in] time_list - list of recorded times
% @param[in] name - name to print info of
time_struct.mean = mean(time_list);
time_struct.stdev = std(time_list);
time_struct.count = length(time_list);
time_struct.min = min(time_list);
time_struct.max = max(time_list);
time_struct.range = max(time_list)-min(time_list);
fprintf("%s : \n",name)
fprintf("    mean : %f\n",time_struct.mean);
fprintf("    stdev : %f\n",time_struct.stdev);
fprintf("    count : %f\n",time_struct.count);
fprintf("    min : %f\n",time_struct.min);
fprintf("    max : %f\n",time_struct.max);
fprintf("    range : %f\n",time_struct.range);
end

function [] = write_json(data,fpath)
% @brief prettyprint data to json file at fpath
jstr = jsonencode(data);
jstr = strrep(jstr, ',', sprintf(',\r'));
jstr = strrep(jstr, '[{', sprintf('[\r{\r'));
jstr = strrep(jstr, '}]', sprintf('\r}\r]'));
fp = fopen(fpath,'w+');
fprintf(fp,jstr);
fclose(fp);
end

%% other functions
function [rv] = generate_random_sparse(m,n,num_el)
   % @brief fill a sparse array with numel_random elements
    data = rand(1,num_el)+1i*rand(1,num_el);
    ri = randi(n,1,num_el);
    ci = randi(n,1,num_el);
    rv = sparse(ri,ci,data,m,n);
end 





