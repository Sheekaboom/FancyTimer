% take our plotly plots and translate them to MATLAB
% this requires tools from https://github.com/Sheekaboom/WeissTools

in_dir = './figs/plotly/json';
out_dir = './figs/matlab';

%get the file names
%dir_vals = dir(fullfile(in_dir,'beamforming*.json'));
dir_vals = dir(fullfile(in_dir,'*.json'));
fnames = {dir_vals.name}; %get the names
%get the file paths
fpaths = fullfile(in_dir,fnames); %full relative file path

%now loop through and save out
for i=1:length(fpaths)
    fpath = fpaths{i};
    [~,name] = fileparts(fpath); %get the name
    fig = plotly2fig(fpath); %load the figure from plotly
    %now save it out
    save_plot(fig,name,out_dir);
end

