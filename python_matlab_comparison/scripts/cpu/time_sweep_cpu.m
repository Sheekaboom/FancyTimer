%% run all matlab timing

file_dir = fileparts(mlfilename('fullpath')); %get the directory of this file
pycom_path = fullfile(file_dir,'../../'); %assume the base is 2 levels up from this

DEBUG = false;

fprintf("Running Basic Sweep");
time_sweep_basic_cpu;
fprintf("Running Extended Sweep");
time_sweep_extended_cpu;
fprintf("Running Matrix Sweep");
time_sweep_matrix_cpu;
fprintf("Running Sparse Sweep");
time_sweep_sparse_cpu;
fprintf("Running FDFD Sweep");
time_sweep_fdfd_cpu;
fprintf("Running Beamforming Sweep");
time_sweep_beamforming_cpu;