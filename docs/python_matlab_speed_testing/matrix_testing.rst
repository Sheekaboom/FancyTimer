
Matrix Operations
=====================
This covers decomposition, multiplication, and solving of linear systems.
- Matrix Multiplication : :math:`C=A*B`
- LU Decomposition : :math:`C=lu(A)`
- Dense Matrix Solve : :math:`A\backslash b`
- Sparse Matrix Solve : :math:`A\backslash b`

Matrix Multiplication
-------------------------
.. math:: C=A*B

CPU
+++++++++
.. raw:: html
	:file: figs/matmul.html

LU Decomposition
--------------------
.. math:: C=lu(A)

CPU
+++++++++
.. raw:: html
	:file: figs/lu.html

Dense Matrix Solving
-----------------------
.. math:: A\backslash b

CPU
++++++++++++
.. raw:: html
	:file: figs/solve.html

Sparse Matrix Solving
-----------------------
.. math:: A\backslash b

CPU
++++++++++++
.. raw:: html
	:file: figs/ssolve.html
    
Code
---------
This was generated with the following scripts:

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
    import numpy as np
    import scipy.linalg

    #%% Some initialization
    mat_dim_list = np.floor(np.linspace(1,10000,51)).astype(np.uint32);
    #mat_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32);
    out_file = 'stats_py_np_matrix.mat'
    #%%
    '''
    NUMPY TESTS
    '''
    #%% Test the fancy_timeit_matrix_sweep capability
    #lu_factor is used as opposed to lu to match the output of MATLAB's lu() function
    functs      = [scipy.linalg.lu_factor,np.matmul]
    funct_names = ['lu'                  ,'matmul' ]
    num_args    = [1                     ,2     ]
    num_reps    = 25

    #% now run 
    [stats_np_double,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.cdouble);
    [stats_np_single,_] = fancy_timeit_matrix_sweep(
        functs,funct_names,num_args,mat_dim_list,num_reps,dtype=np.csingle);

    #%% and now matrix solving 
    import scipy.linalg
    solve_dim_list = np.floor(np.linspace(1,5000,51)).astype(np.uint32)
    #solve_dim_list = np.floor(np.linspace(1,500,6)).astype(np.uint32)
    solve_reps = 25

    #argument generation functions
    def solve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
        return [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.cdouble),
                 (np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)]
    def solve_arg_gen_funct_single(dim,num_args): #generate A and b for dense solving
        return [(np.random.rand(dim,dim)+1j*np.random.rand(dim,dim)).astype(np.csingle),
                 (np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)]
                 
    [solve_stats_double,_] = fancy_timeit_matrix_sweep(
            [scipy.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
            ,arg_gen_funct=solve_arg_gen_funct);
    [solve_stats_single,_] = fancy_timeit_matrix_sweep(
            [scipy.linalg.solve],['solve'],[2],solve_dim_list,solve_reps
            ,arg_gen_funct=solve_arg_gen_funct_single);
            
    stats_np_double.update(solve_stats_double)
    stats_np_single.update(solve_stats_single)

    #%% now save out in case sparse does something stupid
    import scipy.io as sio
    sio.savemat('no_sparse_'+out_file,{'single':stats_np_single,'double':stats_np_double})

    #%% and sparse solving
    import scipy.sparse
    import scipy.sparse.linalg
    import numpy as np

    ssolve_dim_list = np.concatenate(([2],np.linspace(0,5000,51)[1::2])).astype(np.uint32)
    #ssolve_dim_list = np.concatenate(([2],np.linspace(0,500,6)[1::2])).astype(np.uint32)
    ssolve_reps = 25

    def generate_random_sparse(shape,num_el,dtype):
        '''@brief fill a sparse array with numel_random elements'''
        data = (np.random.rand(num_el)+1j*np.random.rand(num_el)).astype(dtype)
        ri = np.random.randint(0,high=shape[0]-1,size=num_el)
        ci = np.random.randint(0,high=shape[1]-1,size=num_el)
        rv = scipy.sparse.csr_matrix((data,(ri,ci)),shape=shape,dtype=dtype)
        return rv  

    #argument generation
    def ssolve_arg_gen_funct(dim,num_args): #generate A and b for dense solving
        numel = np.floor(dim*5).astype(np.uint32)
        return [generate_random_sparse((dim,dim),numel,np.cdouble),(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.cdouble)]  
    def ssolve_arg_gen_funct_single(dim,num_args): #generate A and b for dense solving
        numel = np.floor(dim*5).astype(np.uint32)
        return [generate_random_sparse((dim,dim),numel,np.csingle),(np.random.rand(dim)+1j*np.random.rand(dim)).astype(np.csingle)]  

    #now get the results
    [ssolve_stats_double,_] = fancy_timeit_matrix_sweep(
            [scipy.sparse.linalg.spsolve],['ssolve'],[2],ssolve_dim_list,ssolve_reps
            ,arg_gen_funct=ssolve_arg_gen_funct);
    #now get the results
    [ssolve_stats_single,_] = fancy_timeit_matrix_sweep(
            [scipy.sparse.linalg.spsolve],['ssolve'],[2],ssolve_dim_list,ssolve_reps
            ,arg_gen_funct=ssolve_arg_gen_funct);
            
    stats_np_double.update(ssolve_stats_double)
    stats_np_single.update(ssolve_stats_single)
        
    #%% now save out
    import scipy.io as sio
    sio.savemat(out_file,{'single':stats_np_single,'double':stats_np_double})
        
        
MATLAB
++++++++++

**CPU Code**

.. code-block:: matlab

    addpath('C:\Users\aweis\git\pycom\base');

    %% Some initialization
    mat_dim_list = floor(linspace(1,10000,51));
    %mat_dim_list = floor(linspace(1,500,6));
    out_file = 'stats_mat_matrix.mat';

    %% Test the fancy_timeit_matrix_sweep capability
    %lu_factor is used as opposed to lu to match the output of MATLAB's lu() function
    functs      = {@lu   ,@mtimes };
    funct_names = {'lu'  ,'matmul'};
    num_args    = [1     ,2       ];
    num_reps    = 25;

    % now run 
    stats_mat_double = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@double);
    stats_mat_single = OperationTimer.fancy_timeit_matrix_sweep(...
        functs,funct_names,num_args,mat_dim_list,num_reps,'dtype',@single);

    %% and now matrix solving 
    solve_dim_list = floor(linspace(1,5000,51));
    %solve_dim_list = floor(linspace(1,500,6));
    solve_reps = 25;
                 
    solve_stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
            {@mldivide},{'solve'},[2],solve_dim_list,solve_reps...
            ,'arg_gen_funct',@solve_arg_gen_funct);
    solve_stats_single = OperationTimer.fancy_timeit_matrix_sweep(...
            {@mldivide},{'solve'},[2],solve_dim_list,solve_reps...
            ,'arg_gen_funct',@solve_arg_gen_funct_single);
            
    stats_mat_double = update_struct(stats_mat_double,solve_stats_double);
    stats_mat_single = update_struct(stats_mat_single,solve_stats_single);

    %% now save out in case sparse does something stupid
    single = stats_mat_single;
    double = stats_mat_double;
    save(['no_sparse_',out_file],'single','double');
    clear single double

    %% and sparse solving. MATLAB cant do sparse single
    ssolve_dim_list = floor(linspace(0,5000,51));
    %ssolve_dim_list = floor(linspace(0,500,6));
    ssolve_dim_list = [2,ssolve_dim_list(2:2:end)];
    ssolve_reps = 25;

    %now get the results
    ssolve_stats_double = OperationTimer.fancy_timeit_matrix_sweep(...
            {@mldivide},{'ssolve'},[2],ssolve_dim_list,ssolve_reps...
            ,'arg_gen_funct',@ssolve_arg_gen_funct);
            
    stats_mat_double = update_struct(stats_mat_double,ssolve_stats_double);
    ssolve_stats_double.notes = "THIS IS THE SAME DATA AS FOR DOUBLE. MATLAB DOES NOT SUPPORT SINGLE PRECICION SPARSE";
    stats_mat_single = update_struct(stats_mat_single,ssolve_stats_double); %simply save double as single
        
    %% now save out
    single = stats_mat_single;
    double = stats_mat_double;
    save(out_file,'single','double');

    %% solve data functions
    %argument generation functions
    function args = solve_arg_gen_funct(dim,num_args) %generate A and b for dense solving
        args = {};
        args{1} = double(rand(dim,dim)+1i*rand(dim,dim));
        args{2} = double(rand(dim,1)+1i*rand(dim,1));
     end
    function args = solve_arg_gen_funct_single(dim,num_args) %generate A and b for dense solving
        args = {};
        args{1} = single(rand(dim,dim)+1i*rand(dim,dim));
        args{2} = single(rand(dim,1)+1i*rand(dim,1));
     end
             
    %% ssolve data functions
    function spmat = generate_random_sparse(shape,num_el,dtype)
        %@brief fill a sparse array with numel_random elements
        data = rand(1,num_el)+1i*rand(1,num_el);
        ri = randi([1,shape(1)],1,num_el);
        ci = randi([1,shape(2)],1,num_el);
        spmat = dtype(sparse(ri,ci,data,shape(1),shape(2)));
    end

    function args = ssolve_arg_gen_funct(dim,num_args) %generate A and b for dense solving
        numel = floor(dim*5);
        args = {};
        args{1} = generate_random_sparse([dim,dim],numel,@double);
        args{2} = rand(dim,1)+1i*rand(dim,1);
    end

    %% funciton for updating a structure
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
        
        





