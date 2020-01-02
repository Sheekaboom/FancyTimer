    
Timing Code
============
Code was specifically written to provide better statistics on timing for these tests.
An example to run this code is as follows:

Python
+++++++++
.. code-block:: python
 
    #import dependencies
    from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
    import numpy as np

    #set our sweep size (use default argument generation function
    mat_dim_list = np.floor(np.linspace(1,10000,6)).astype(np.uint32);
    out_file_np = 'stats_py_np_basic.mat'

    #now set our list of functions
    combo_fun   = lambda x,y: x+y*np.exp(x)
    functs      = [np.add,np.subtract,np.multiply,np.divide,np.exp,combo_fun]
    funct_names = ['add' ,'sub'      ,'mult'     ,'div'    ,'exp' ,'combo'  ]
    num_args    = [2     ,2          ,2          ,2        ,1     ,2        ]
    num_reps    = 100

    #and run the time tests
    [stats_np_double,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
    [stats_np_single,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);
    
    #then save to matlab *.mat file
    sio.savemat(out_file_np,{'double':stats_np_double,'single':stats_np_single})
   
MATLAB
+++++++++
   
.. code-block:: matlab

    #add the path of our timing class
    addpath('C:\Users\aweis\git\pycom\base');

    #set our sweep size (use default argument generation function
    mat_dim_list = floor(linspace(1,10000,51));
    out_file = 'stats_mat_basic.mat';

    #now set our list of functions
    combo_fun   = @(x,y) x+y*exp(x);
    functs      = {@plus ,@minus,@times ,@rdivide,@exp  ,combo_fun};
    funct_names = {'add' ,'sub' ,'mult' ,'div'   ,'exp' ,'combo'  };
    num_args    = [2    ,2    ,2     ,2      ,1    ,2        ];
    num_reps    = 100;

    #and run the time tests
    stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
    stats_single = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);

    #save out the data
    single = stats_single;
    double = stats_double;
    save(out_file,'single','double');
   
Libraries for Timing 
---------------------

Python
+++++++++++++
.. automodule:: base.OperationTimer
	:members:

MATLAB
++++++++++
Gotta get this to work

mat:autoclass:: base.OperationTimer
:members: