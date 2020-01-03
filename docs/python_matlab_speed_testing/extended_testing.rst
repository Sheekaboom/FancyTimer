
Extended Operations
=====================
- exp  : :math:`c=e^{a}`
- combo : :math:`c=a+b*e^{a}`
- Summation : :math:`sum(a)`
- Fast Fourier Transform : :math:`c=fft(a)`
    
Exponentiation
------------------
.. math:: c=e^{a}

CPU
++++++
.. raw:: html
	:file: figs/exp.html
    
Combination
----------------
.. math:: c=a+b*e^{a}

CPU
++++++
.. raw:: html
	:file: figs/combo.html

Summation
-----------
.. math:: c=sum(a)

CPU
++++++
.. raw:: html
	:file: figs/sum.html

Fast Fourier Transform (FFT) 
-------------------------------   
.. math:: c=fft(a)

.. raw:: html
	:file: figs/fft.html
    
Code
---------
This was generated with the following scripts:

Python
++++++++

**CPU Code**

.. code-block:: python

    """
    Created on Tue Nov  5 16:14:54 2019

    @author: aweiss
    """

    from pycom.base.OperationTimer import fancy_timeit_matrix_sweep
    import numpy as np

    #%% Some initialization
    #mat_dim_list = np.floor(np.linspace(1,1000,6)).astype(np.uint32);
    mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
    out_file = 'stats_py_np_extended.mat'
    #%%
    '''
    NUMPY TESTS
    '''
    #%% Test the fancy_timeit_matrix_sweep capability
    functs      = [np.sum]
    funct_names = ['sum' ]
    num_args    = [1     ]
    num_reps    = 100

    #% now run 
    [stats_np_double,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
    [stats_np_single,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);
        
        
    #%% now measure our fft
    fft_dim_list = np.floor(np.linspace(1,5000000,51)).astype(np.uint32)
    #fft_dim_list = np.floor(np.linspace(1,50000,6)).astype(np.uint32)

    fft_reps = 100

    # generate our arguments
    def fft_arg_gen_funct(dim,num_args): #vector generation function 
            return [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)
                                        for a in range(num_args)]
    # generate our arguments
    def fft_arg_gen_funct_single(dim,num_args): #vector generation function 
            return [(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)
                                        for a in range(num_args)]
        
    [stats_fft_double,_] = fancy_timeit_matrix_sweep(
            [np.fft.fft],['fft'],[1],fft_dim_list,fft_reps
            ,arg_gen_funct=fft_arg_gen_funct);
    stats_np_double.update(stats_fft_double)

    [stats_fft_single,_] = fancy_timeit_matrix_sweep(
            [np.fft.fft],['fft'],[1],fft_dim_list,fft_reps
            ,arg_gen_funct=fft_arg_gen_funct_single);
    stats_np_single.update(stats_fft_single)

        
    #%% now save out
    import scipy.io as sio
    sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})
        

MATLAB
+++++++++

**CPU Code**

.. code-block:: matlab 

    addpath('C:\Users\aweis\git\pycom\base');

    %% Some initialization
    %mat_dim_list = floor(linspace(1,1000,6));
    mat_dim_list = floor(linspace(1,10000,51));
    out_file = 'stats_mat_extended.mat';

    %% Test the fancy_timeit_matrix_sweep capability
    sum_fun = @(x) sum(sum(x));
    functs      = {sum_fun};
    funct_names = {'sum'  };
    num_args    = [1      ];
    num_reps    = 100;

    % now run 
    stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
    stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);
        
        
    %% now measure our fft
    fft_dim_list = floor(linspace(1,5000000,51));
    %fft_dim_list = floor(linspace(1,50000,6));

    fft_reps = 100;
        
    stats_fft_double = OperationTimer.fancy_timeit_matrix_sweep(...
            {@fft},{'fft'},[1],fft_dim_list,fft_reps...
            ,'arg_gen_funct',@fft_arg_gen_funct);
    stats_mat_double = update_struct(stats_mat_double,stats_fft_double);

    stats_fft_single = OperationTimer.fancy_timeit_matrix_sweep(...
            {@fft},{'fft'},[1],fft_dim_list,fft_reps...
            ,'arg_gen_funct',@fft_arg_gen_funct_single);
    stats_mat_single = update_struct(stats_mat_single,stats_fft_single);
        
    %% now save out
    single = stats_mat_single;
    double = stats_mat_double;
    save(out_file,'single','double');
        
    %% functions for fft arg generation                                     
    function args = fft_arg_gen_funct(dim,num_args)
         %@brief generate default arguments
        args = {};
        for argn=1:num_args
            args{end+1} = double(rand(1,dim)+1i*rand(1,dim));
        end
    end

    function args = fft_arg_gen_funct_single(dim,num_args)
         %@brief generate default arguments
        args = {};
        for argn=1:num_args
            args{end+1} = single(rand(1,dim)+1i*rand(1,dim));
        end
    end

    function new_struct = update_struct(struct_to_update,struct_to_add)
        %@brief Add all of the fields of struct_to_add to struct_to_update
        %@note If overlapping fields exist, struct_to_update will be overwritten
        %@param[in] struct_to_update - what struct are we updating
        %@param[in] struct_to_add - struct with fields to add to struct_to_update
        %@return New structure with fields from both input structs
        sta_fldnames = fieldnames(struct_to_add);
        new_struct = struct_to_update; %copy struct to update
        for i=1:length(sta_fldnames) %add all of our fieldnames
            fldname = sta_fldnames{i};
            new_struct.(fldname) = struct_to_add.(fldname); %add to new struct
        end
    end
    
    
    
    
	
	
	
        
        
        





