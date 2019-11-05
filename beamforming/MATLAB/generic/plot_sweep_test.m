%% Plot our sweeps of our data
data_names = {'add','sub','mult','div'};
load('mat_stats.mat');
load('py_stats.mat');
for d=1:length(data_names)
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(mat_stats_1.(dname),[dname,' MATLAB']); %plot matlab data
    plot_timeit_sweep( py_stats_1.(dname),[dname,' Python']); %plot Python data
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('N for NxN matrix'); ylabel('Runtime (seconds)'); legend('show');
    savefig(fig,fullfile('figs',[dname,'_speed_comp.fig']));
end

data_names = {'lu','matmul'};
for d=1:length(data_names)
    dname = data_names{d};
    fig = figure();hold on;grid on; %make a new figure
    plot_timeit_sweep(mat_stats_2.(dname),[dname,' MATLAB']); %plot matlab data
    plot_timeit_sweep( py_stats_2.(dname),[dname,' Python]']); %plot Python data
    title(sprintf('Runtimes MATLAB vs. Python for %s',dname));
    xlabel('N for NxN matrix'); ylabel('Runtime (seconds)'); legend('show');
    savefig(fig,fullfile('figs',[dname,'_speed_comp.fig']));
end
    