FDFD Simulation
==================
Here a finite difference frequency domain (FDFD) problem was solved.
Run times between MATLAB and python for a 2D cylinder scattering problem were compared. 

Results
-----------
The output of this problem 
130 cells in the x and y direction look as follows:

.. raw:: html
	:file: figs/FDFD_results.html

Speed Comparison
--------------------
Speed comparisons between python and MATLAB can be seen below:

CPU
+++++++++
.. raw:: html
    :file: figs/fdfd.html
	
	
Code
-----------

The following class was used to calculate the 2D FDFD solution:

.. automodule:: python_matlab_comparison.FDFD.FDFD_2D
	:members:
    
The runtimes were generated with the following scripts:

Python
+++++++++

**CPU Code**

.. code-block:: python 

    from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
    from pycom.python_matlab_comparison.FDFD.FDFD_2D import FDFD_2D
    import numpy as np
    import scipy.linalg

    #%% Some initialization
    dim_list = np.floor(np.linspace(1,500,51)).astype(np.uint32);
    #dim_list = np.floor(np.linspace(1,50,6)).astype(np.uint32);
    out_file = 'stats_py_np_fdfd.mat'

    num_reps = 100
    #num_reps = 10

    stats_np_single = {}
    stats_np_double = {}

    #argument generation
    def fdfd_arg_gen_funct(dim,num_args): #generate num_cells_x,num_cells_y as dim
        return [dim,dim,np.cdouble]  
    def fdfd_arg_gen_funct_single(dim,num_args): #generate num_cells_x,num_cells_y as dim
        return [dim,dim,np.csingle]  

    #now run
    [solve_stats_double,_] = fancy_timeit_matrix_sweep(
            [FDFD_2D],['fdfd'],[3],dim_list,num_reps
            ,arg_gen_funct=fdfd_arg_gen_funct);
    [solve_stats_single,_] = fancy_timeit_matrix_sweep(
            [scipy.linalg.solve],['fdfd'],[3],dim_list,num_reps
            ,arg_gen_funct=fdfd_arg_gen_funct_single);
            
    stats_np_double.update(solve_stats_double)
    stats_np_single.update(solve_stats_single)
        
    #%% now save out (single is not used)
    import scipy.io as sio
    sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})




MATLAB
++++++++++++

**CPU Code**

.. code-block:: matlab

    path_to_pycom = 'C:\Users\aweis\git\';
    addpath(fullfile(path_to_pycom,'pycom\python_matlab_comparison\FDFD\'));
    addpath(fullfile(path_to_pycom,'pycom\base'));

    %% Some initialization
    %dim_list = floor(linspace(1,50,6));
    dim_list = floor(linspace(1,500,51));
    out_file = 'stats_mat_fdfd.mat';

    %% Test the fancy_timeit_matrix_sweep capability
    functs      = {@FDFD_2D};
    funct_names = {'fdfd'  };
    num_args    = [2      ];
    num_reps    = 100;
    %num_reps = 5;

    % now run 
    stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,dim_list,num_reps,'arg_gen_funct',@fdfd_arg_gen_funct);
        
    %% now save out
    m_single = struct(); % matlab cant do sparse single
    m_double = stats_mat_double;
    save(out_file,'m_single','m_double');
        
    %% functions for fft arg generation                                     
    function args = fdfd_arg_gen_funct(dim,num_args)
         %@brief generate default arguments
        args = {dim,dim};
    end






