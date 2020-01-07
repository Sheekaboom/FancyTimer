Beamforming Simulation
========================================

Beamforming was tested with a synthetic 35x35 element array at 40GHz each element was spaced
int he x and y directions by :math:`\frac{\lambda}{2}` calculated at 40 GHz. For the speed
results, the number of angles calculated were swept from 1 to 500 angles in both the azimuthal
and elevation dimensions. These different angles were then used to create a meshgrid providing
a total number of calculated angles as the input to :func:`pycom.python_matlab_comparison.beamforming.beamform_speed.beamform_speed`
as num_angles^2. All tests were run with a single incident plane wave simulated at [pi/4,-pi/4].

Output results
------------------
181 angles (1 degree steps) in both azimuth and elevation provides the following 3D output.
This is done with a single incident plane wave simulated at [pi/4,-pi/4].


.. raw:: html
	:file: figs/beamform_results.html


Speed Comparison
--------------------
Speed comparisons between python and MATLAB can be seen below:
    
CPU
+++++++++

.. raw:: html
    :file: figs/beamforming.html


GPU
+++++++++
results


Code 
-------

The following module was used to calculate the Beamforming solution:

.. automodule:: python_matlab_comparison.beamforming.beamform_speed
	:members:
    
The runtimes were generated with the following scripts:

Python
+++++++++

**CPU Code** 

.. code-block:: python 

    # -*- coding: utf-8 -*-
    """
    Created on Tue Nov  5 16:14:54 2019

    @author: aweiss
    """

    from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
    from pycom.python_matlab_comparison.beamforming.beamform_speed import beamform_speed
    import numpy as np
    import scipy.linalg

    #%% Some initialization
    dim_list = np.floor(np.linspace(1,500,51)).astype(np.uint32);
    #dim_list = np.floor(np.linspace(1,50,6)).astype(np.uint32);
    out_file = 'stats_py_np_beamforming.mat'

    #num_reps = 100
    num_reps = 10

    stats_np_single = {}
    stats_np_double = {}

    #argument generation
    def beamforming_arg_gen_funct(dim,num_args): #generate num_cells_x,num_cells_y as dim
        return [dim,np.cdouble]  
    def beamforming_arg_gen_funct_single(dim,num_args): #generate num_cells_x,num_cells_y as dim
        return [dim,np.csingle]  

    #now run
    [solve_stats_double,_] = fancy_timeit_matrix_sweep(
            [beamform_speed],['beamforming'],[2],dim_list,num_reps
            ,arg_gen_funct=beamforming_arg_gen_funct);
    [solve_stats_single,_] = fancy_timeit_matrix_sweep(
            [beamform_speed],['beamforming'],[2],dim_list,num_reps
            ,arg_gen_funct=beamforming_arg_gen_funct_single);
            
    stats_np_double.update(solve_stats_double)
    stats_np_single.update(solve_stats_single)
        
    #%% now save out (single is not used)
    import scipy.io as sio
    sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})



MATLAB
+++++++++

**CPU Code**

.. code-block:: matlab

    path_to_pycom = 'C:\Users\aweis\git\';
    addpath(fullfile(path_to_pycom,'pycom\python_matlab_comparison\beamforming\'));
    addpath(fullfile(path_to_pycom,'pycom\base'));

    %% Some initialization
    dim_list = floor(linspace(1,50,6));
    %dim_list = floor(linspace(1,500,51));
    out_file = 'stats_mat_beamforming.mat';

    %% Test the fancy_timeit_matrix_sweep capability
    functs      = {@beamform_speed};
    funct_names = {'beamforming'  };
    num_args    = [2      ];
    num_reps    = 100;
    num_reps = 5;

    % now run 
    stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct);
    stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct_single);
        
    %% now save out
    m_single = stats_mat_single; % matlab cant do sparse single
    m_double = stats_mat_double;
    save(out_file,'m_single','m_double');
        
    %% functions for fft arg generation                                     
    function args = fdfd_arg_gen_funct(dim,num_args)
         %@brief generate default arguments
        args = {dim,@double};
    end

    function args = fdfd_arg_gen_funct_single(dim,num_args)
         %@brief generate default arguments
        args = {dim,@single};
    end
	



