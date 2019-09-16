'''
@author ajw
@date 8-24-2019
@brief functions and classes for creating
    python functions from fortran subroutines
'''

class BeamFort:
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    def __init__(self):
        '''
        @brief constructor
        '''
        pass
    

# deprecated. just call libraries after building .so files using cmake
def build_fortran_library(file_path,module_name=None,out_subdir='python_bindings'):
    '''
    @brief compile fortran code to a module. The code may need to be compiled
        With CMAKE first to ensure any needed other files are built
    @param[in] file_path - path to fortran file to build code from
    @param[in/OPT] module_name - name to output module to. If none use the file name
    @param[in/OPT] out_subdir - subdirectory to save the files into
    '''
    from numpy import f2py
    #first get the name of the input file for the output name
    wdir,fname = os.path.split(file_path)
    if module_name is None:
        module_name,_ = os.path.splitext(fname)
    out_dir = os.path.join(wdir,out_subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    f2pyrv = f2py.run_main(['-c',file_path,'--build-dir',out_dir,'-m',module_name])
    return f2pyrv
    
if __name__=='__main__':
    import os
    import ctypes
    wdir = r'C:\Users\aweis\git\pycom\beamforming\fortran\build'
    mod = ctypes.CDLL(os.path.join(wdir,'libbeamform_serial.dll'))
    mod.__beamforming_serial_MOD_fortran_mult_funct.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
    mod.__beamforming_serial_MOD_fortran_mult_funct.restype = ctypes.c_float
    ca = ctypes.c_float(3.)
    cb = ctypes.c_float(5.)
    res = mod.__beamforming_serial_MOD_fortran_mult_funct(ctypes.pointer(ca),ctypes.pointer(cb))
    print("{} * {} = {}".format(ca,cb,res))
    
    
    