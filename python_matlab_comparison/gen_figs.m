% Plot our sweeps of our data for CPU data
clear all;
load('stats_mat_11-10-2019.mat');
load('stats_py_11-10-2019.mat');
data_names = {'add','sub','mult','div'};
np_dname = 'Numpy';
nb_dname = 'Numba';
mt_dname = 'MATLAB';
% for d=1:length(data_names)
%     cidx = 1; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('sqrt(N)'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
%     
% end
% 
% data_names = {'exp','combo'};
% for d=1:length(data_names)
%     cidx = 2; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('sqrt(N)'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
% end

data_names = {'sum'}; %separate because Numba doesnt have sum
for d=1:length(data_names) 
    cidx = 2; %cell in structure we store this data
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
    plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('sqrt(N)','Interpreter','latex'); ylabel('Runtime (seconds)'); legend('show');
    save_plot(fig,[dname,'_speed_comp'],'figs');
    mc_fig = figure(); hold on; grid on; %matlab comparison figure
    plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
    %plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
    xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
    save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
end

data_names = {'lu','matmul'};
for d=1:length(data_names)
    cidx = 3; %cell in structure we store this data
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
    plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
    %plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('sqrt(N)'); ylabel('Runtime (seconds)'); legend('show');
    save_plot(fig,[dname,'_speed_comp'],'figs');
    mc_fig = figure(); hold on; grid on; %matlab comparison figure
    plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
    xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
    save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
end


%%Sparse and Dense Solving
load('solve_stats_mat.mat')
load('solve_stats_py.mat')
data_names = {'solve'};
for d=1:length(data_names)
    cidx = 1; %cell in structure we store this data
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
    plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
    %plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('sqrt(N)'); ylabel('Runtime (seconds)'); legend('show');
    save_plot(fig,[dname,'_speed_comp'],'figs');
    mc_fig = figure(); hold on; grid on; %matlab comparison figure
    plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
    xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
    save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
end

data_names = {'ssolve'};
for d=1:length(data_names)
    cidx = 2; %cell in structure we store this data
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
    plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
    %plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('sqrt(N)'); ylabel('Runtime (seconds)'); legend('show');
    save_plot(fig,[dname,'_speed_comp'],'figs');
    mc_fig = figure(); hold on; grid on; %matlab comparison figure
    plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
    xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
    save_plot(mc_fig,[dname,'_speed_comp_rel_matlab'],'figs');
end

% FFT CPU plots %where is this data???
%load('fft_stats_mat.mat')
%load('solve_stats_py.mat')
data_names = {'fft'};

%% now lets run our gpu stuff
np_dname = 'Pycuda/Skcuda';
nb_dname = 'CuPy';
mt_dname = 'MATLAB';
% 
% load('stats_py_gpu.mat');
% load('stats_mat_gpu_initial.mat');
% stats_mat = stats_gpu;
% data_names = {'add','sub','mult','div'};
% for d=1:length(data_names)
%     cidx = 1; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('Number of Elements'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end
% 
% data_names = {'exp','combo'};
% for d=1:length(data_names)
%     cidx = 2; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('Number of Elements'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end
% 
% data_names = {'sum'}; %separate because Numba doesnt have sum
% for d=1:length(data_names) 
%     cidx = 2; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end
% 
% load('stats_mat_gpu_initial.mat');
% stats_mat = stats_gpu;
% %%Missing MATMUL (does seem to have saved)
% data_names = {'matmul'};
% for d=1:length(data_names)
%     cidx = 1; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('Number of Elements'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end
% 
% %%dense solving
% data_names = {'fft'};
% for d=1:length(data_names)
%     cidx = 2; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('Number of Elements'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end
% 
% %%dense solving
% data_names = {'solve'};
% for d=1:length(data_names)
%     cidx = 4; %cell in structure we store this data
%     dname = data_names{d};
%     fig = figure();hold on;grid on; %make a new figure
%     plot_timeit_sweep(stats_mat{cidx}.(dname),[dname,' ',mt_dname]); %plot matlab data
%     plot_timeit_sweep(stats_py_np{cidx}.(dname),[dname,' ',np_dname]); %plot Python data numpy
%     %plot_timeit_sweep(stats_py_nb{cidx}.(dname),[dname,' ',nb_dname]); %plot Python data numba
%     title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
%     xlabel('Number of Elements'); ylabel('Runtime (seconds)'); legend('show');
%     save_plot(fig,[dname,'_speed_comp_gpu'],'figs');
%     mc_fig = figure(); hold on; grid on; %matlab comparison figure
%     plot_timeit_sweep_reference(stats_py_np{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',np_dname]);
%     %plot_timeit_sweep_reference(stats_py_nb{cidx}.(dname),stats_mat{cidx}.(dname),[dname,' ',nb_dname]);
%     xlabel('sqrt(N)','Interpreter','latex'); ylabel('Relative runtime to MATLAB'); legend('show');
%     save_plot(mc_fig,[dname,'_speed_comp_rel_matlab_gpu'],'figs');
% end

%% What is needed
% - python CPU fft
% - gpu matmul python
% - reorder data to align between python and MATLAB
    