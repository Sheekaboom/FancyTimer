function [fig_handle] = plot_timeit_sweep(stats,name)
%PLOT_TIMEIT_SWEEP Summary of this function goes here
%   Detailed explanation goes here
fnames = fieldnames(stats); %get our field names
size_m = zeros(1,length(fnames)); %size of 1 side of the matrix
means  = zeros(1,length(fnames)); %mean of our times
stdev  = zeros(1,length(fnames)); %stdev value
mins   = zeros(1,length(fnames)); %stdev value
maxs   = zeros(1,length(fnames)); %stdev value
for i=1:length(fnames)
    %first extract the matrix size. This has m_ prepended to it to 
    %       work with MATLAB struct syntax
    size_m(i) = str2num(regexprep(fnames{i},'[^0-9]*',''));
end
%now sort the sizes and extract the data
size_m = sort(size_m);
for i=1:length(size_m)
    %now extract our data
    fname = ['m_',num2str(size_m(i))]; %reconstruct from sorted data
    means(i) = stats.(fname).mean;
    stdev(i) = stats.(fname).stdev;
    mins(i)  = stats.(fname).min;
    maxs(i)  = stats.(fname).max;
end
%stdev_u = means+stdev;
%stdev_l = means-stdev;
fig_handle = gcf();
%plot(size_m,means,'-+','DisplayName',[name,': Mean']);hold on;
%plot(size_m,stdev_u,'DisplayName',[name,': Std. +']);
%plot(size_m,stdev_l,'DisplayName',[name,': Std. -']);
errorbar(size_m,means,stdev,'DisplayName',[name]);
end

