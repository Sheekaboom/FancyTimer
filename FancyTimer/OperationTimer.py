# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:16:02 2019

@author: aweiss
"""
import timeit
import numpy as np
import inspect
import re
import warnings

try:
    from WeissTools.Dict import WDict
except ModuleNotFoundError:
    from collections import OrderedDict as WDict       
  
class FancyTimerStats(WDict):
    '''
    @brief class to hold and manipulate data from fancy_timeit function
    '''
    def __init__(self,time_list,*args,**kwargs):
        '''
        @brief constructor
        @param[in] time_list - list of times that were measured for our repeats
        '''
        super().__init__(*args,**kwargs)
        self._calc_stats(time_list)
        
    def _calc_stats(self,time_list):
        '''@brief calculate and store time stats'''
        time_list = np.array(time_list)
        self['mean'] = np.mean(time_list)
        self['stdev']= np.std(time_list)
        self['count']= len(time_list)
        self['min']  = np.min(time_list)
        self['max']  = np.max(time_list)
        self['range']= np.ptp(time_list)
        self['raw'] = time_list
        
    def print(self):
        '''
        @brief print our time data from our Function
        '''
        for k,v in self.items():
            print('    {} : {}'.format(k,v))
        
    def set_speedup(self,base_fts):
        '''
        @brief add speedup statistics to the timer
        @param[in] base_fts - baseline FancyTimerStats class 
        '''
        self['speedup'] = base_fts['mean']/self['mean']
        
class FancyTimerStatsSet(WDict):
    '''
    @brief a class to hold a collection of FancyTimerStats.
    '''
    def __init__(self,*args,**kwargs):
        '''
        @brief constructor
        @param[in] Same inputs as a regular dictionary
        '''
        super().__init__(*args,**kwargs)
        
    def get_stats_as_arrays(self,sort=True):
        '''
        @brief Get all of the statistics as arrays for operating on
        @param[in/OPT] sort - whether or not to sort by size (True)
        @return A dictionary with keys 'mean','stdev','min','max', and 'size'
            each mapping to an array of the corresponding data
        '''
        array_dict = {}
        array_dict['size']  = np.array([int(re.sub('[^0-9]*','',k)) for k in self.keys()])
        stats_names = list(self.values())[0].keys()
        for stat_name in stats_names:
            array_dict[stat_name] = np.array([stat[stat_name] for stat in self.values()])
        if sort:
            sort_idx = np.argsort(array_dict['size'])
            for k,v in array_dict.items(): #now sort each value (including size)
                array_dict[k] = v[sort_idx]
        return array_dict
        
class FancyTimerStatsMatrixSet(FancyTimerStatsSet):
    '''
    @brief class to hold a collection of FancyTimerStats for a matrix test
        Each key should be the size of 1 dimension of a 2D square matrix
    '''
    def __init__(self,*args,**kwargs):
        '''
        @brief constructor
        @param[in] Same inputs as a regular dictionary
        '''
        super().__init__(*args,**kwargs)   

def fancy_timeit(mycallable,num_reps=3,num_calls=1,**kwargs):
    '''
    @brief easy timeit function that will return a dictionary of timing statistics
    @param[in] mycallable - callable statement to time
    @param[in] num_calls - number of calls per time. This is useful for functions 
        that finish really quickly, although they cannot be used for statistics
    @param[in/OPT] num_reps - number of repeats for timing and statistics
    @return A FancyTimerStats object of timing statistics of mycallable
    '''
    fancy_template = '''
def inner(_it, _timer{init}):
    {setup}
    time_list = []
    for _i in _it:
        _t0 = _timer()
        #retval = {stmt}
        rv = {stmt}
        _t1 = _timer()
        time_list.append((_t1-_t0)/len(rv)) #append the time to run
    return time_list #, retval
'''
    
    #if num_calls>1: mycallable_str = '[{call} for i in range({num_calls})]'.format(call=mycallable,num_calls=num_calls)
    #else: mycallable_str = '{call}'.format(call=mycallable)
    stmt_rep_str = '[{stmt} for i in range(%d)]' %(num_calls)
    fancy_template = fancy_template.replace('{stmt}',stmt_rep_str)
    # now run the timeit stuff
    timeit.template = fancy_template #set the template
    ft = timeit.Timer(stmt=mycallable,**kwargs)
    #tl,rv = ft.timeit(number=num_reps)
    tl = ft.timeit(number=num_reps)
    return FancyTimerStats(tl)

def display_time_stats(time_data,name):
    '''
    @brief print our time data from our Function
    '''
    print('{} :'.format(name))
    time_data.print()
    return time_data 

def fancy_timeit_sweep(functs:list,args:list,num_reps:list,num_calls:list=1,kwargs:list=None,verbose=True):
    '''
    @brief time a list of functs each for a given list of args a number of reps specified 
        by the corresponding entry in num_reps and same for num_calls (see fancy_timeit)
    @param[in] functs - iterable of functions to time. Each should have the same number of inputs
    @param[in] args - list of tuples for swept arguments to pass to each function
    @param[in] num_reps - list of number of repeat arguments to run for each funct and arg
    @param[in] num_calls - list of integers. See fancy_timeit comments
    @param[in/OPT] kwargs - list of dicts to pass as kwargs
    @return dict with function names as keys and list of FancyTimerStats for each corresponding args
    '''
    # clean input
    if np.isscalar(num_reps): num_reps = [num_reps]*len(args)
    if np.isscalar(num_calls):num_calls = [num_calls]*len(args)
    if isinstance(args,tuple): args = [args]
    if kwargs is None: kwargs = {}
    if isinstance(kwargs,dict): # allow passing single kwargs for all
        kwargs = [kwargs]*len(args)
    # now lets test
    out_stats = WDict()
    for funct in functs: # go through each function
        fname = str(funct) # get the name
        if verbose: print("Fancily Timing {} 🧐".format(fname))
        fstats = []
        for i in range(len(args)):
            try: # try and push through if we have any errors
                if verbose>1:print("Running for args[{}]".format(i))
                cur_stat = fancy_timeit(lambda: funct(*args[i],**kwargs[i]),num_reps=num_reps[i])
                fstats.append(cur_stat)
            except Exception as e:
                warnings.warn("EXCEPTION THROWN: {}".format(e))
                fstats.append(str(e))
        out_stats[fname] = fstats 
    return out_stats

def fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,dim_range,num_reps,num_calls=1,**kwargs):
    '''
    @brief easy timeit function that will return the results of a function
        along with a dictionary of timing statistics
    @param[in] funct_list - list of function handles to run
    @param[in] funct_names - list of names for storing results
            Explicitly providing these provides uniformity between libs
    @param[in] num_arg_list - list of the number of input arguments (matrices)
            that should be passed to the corresponding function in myfunct_list
    @param[in] dim_range - range of numbers to use as M for an MxM matrix
    @param[in/OPT] num_reps - number of repeats for timing and statistics
    @param[in/OPT] kwargs - keyword arguments as follows:
        - arg_gen_funct - function to generate the matrix from a dim value input
            if not included default to generate np.cdouble random matrix
        - timer_funct - function to use for timing. If not included uses fancy_timeit
        - dtype - dtype to use for the default arg_gen_funct. should be cdouble or csingle
        - cleanup_funct - function to run on arg_inputs after each dimension iteration
            must recieve list of args to cleanup
    '''
    def default_cleanup(arg_list):
        pass
    
    options = {}
    options['dtype'] = np.cdouble 
    options['cleanup_funct'] = default_cleanup
    options['no_fail'] = False #whether or not exceptions will be passed
    for k,v in kwargs.items(): #we parse this first to get dtype
        options[k] = v 
    #then we generate the default function
    def arg_gen_funct(dim,num_args): #default matrix generation
        return [options['dtype'](np.random.rand(dim,dim)+1j*np.random.rand(dim,dim))
                                                                        for a in range(num_args)]
    #then we parse again
    options['arg_gen_funct'] = arg_gen_funct
    options['timer_funct'] = fancy_timeit
    for k,v in kwargs.items():
        options[k] = v    
    #now lets get our function names as strings
    #now create our statistics dictionary
    stats = {fn:FancyTimerStatsMatrixSet() for fn in funct_names}
    #ret_vals = {}
    max_num_args = np.max(num_arg_list) #get the maximum number of arguments
    #now loop through each size of our matrix
    for dim in dim_range: #loop through each of
        #first create our random matrices
        rv = None
        if not options['no_fail']: #allow faliure
            arg_inputs = options['arg_gen_funct'](dim,max_num_args)
            if np.ndim(arg_inputs[0])==0:
                print("Running with input of {}, dtype={}:".format(arg_inputs[0],arg_inputs[0].dtype))
            else:
                print("Running with matrix of {}, dtype={}:".format(np.shape(arg_inputs[0]),arg_inputs[0].dtype))
            #now run each function
            for i,funct in enumerate(funct_list):   
                funct_name = funct_names[i]
                num_args = num_arg_list[i]
                print('    %10s :' %(funct_name),end='');
                lam_fun = lambda: funct(*tuple(arg_inputs[:num_args]))
                cur_stat = options['timer_funct'](lam_fun,num_reps=num_reps,num_calls=num_calls)
                stats[funct_name]['m_'+str(dim)] = cur_stat
                #ret_vals[funct_name] = rv
                print(' SUCCESS')
                rv = None #try to clear memory
            options['cleanup_funct'](arg_inputs) #pass in list of args
        else:
            while True:
                try:
                    #first create our random matrices
                    arg_inputs = options['arg_gen_funct'](dim,max_num_args)
                    print("Running with matrix of {}, dtype={}:".format(np.shape(arg_inputs[0]),arg_inputs[0].dtype))
                    #now run each function
                    for i,funct in enumerate(funct_list):   
                        funct_name = funct_names[i]
                        num_args = num_arg_list[i]
                        print('    %10s :' %(funct_name),end='');
                        lam_fun = lambda: funct(*tuple(arg_inputs[:num_args]))
                        cur_stat = options['timer_funct'](lam_fun,num_reps=num_reps)
                        stats[funct_name]['m_'+str(dim)] = cur_stat
                        #ret_vals[funct_name] = rv
                        print(' SUCCESS')
                        rv = None #try to clear memory
                    options['cleanup_funct'](arg_inputs) #pass in list of args
                except BaseException as e:
                    print("Failure with exception {}".format(e.__class__.__name__))
                    if e.__class__ is KeyboardInterrupt: raise KeyboardInterrupt #allow keyboard interrupt
                    continue
                break
    return stats,rv
      
#%% some testing  

import unittest

class TestOperationTimer(unittest.TestCase):
    '''@brief test the Operation Timer class'''
    
    def test_matmul_timing(self):
        '''@brief test timing using numpy matmul with a.size=(10000,1000), b.size=(1000,1000)'''
        print("Testing")
        a = np.random.rand(10000,1000)
        b = np.random.rand(1000,1000)
        times = fancy_timeit(lambda: a@b,10,100)
        print(times)
        times2 = fancy_timeit(lambda: a@b,10,1)
        print(times2)


if __name__=='__main__':
    
    unittest.main()
    
    #%% Test the fancy_timeit_matrix_sweep capability
    #mat_dim_list = 2**np.arange(4,14);
    #mat_dim_list = np.floor(np.logspace(1,4,100)).astype(np.int32)
    '''
    mat_dim_list = 2**np.arange(4,11);
    funct_list   = [np.add,np.subtract,np.multiply,np.divide];
    funct_names  = ['add' ,'sub'      ,'mult'     ,'div'    ];
    num_arg_list = [2     ,2          ,2          ,2        ];
    [py_stats_1,rv] = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,100,dtype=np.csingle);
    '''
    '''
    from scipy.linalg import lu
    funct_list   = [lu  ,np.matmul]
    funct_names  = ['lu','matmul' ]
    num_arg_list = [1   ,2        ]
    [py_stats_2,rv] = fancy_timeit_matrix_sweep(funct_list,funct_names,num_arg_list,mat_dim_list,10);
    
    #%% write out to matlab
    import scipy.io as sio
    sio.savemat('py_stats.mat',{'py_stats_1':py_stats_1,'py_stats_2':py_stats_2})
    '''
        
        




