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
    
    
def compile_fortran_code(code_path):
    '''
    @brief compile fortran code to a module. The code may need to be compiled
        With CMAKE first to ensure any needed other files are built
    '''
    from numpy import f2py
    #first get the name of the input file for the output name
    wdir,fname = os.path.split(code_path)
    fname,_ = os.path.splitext(fname)
    out_path = os.path.join(wdir,fname)
    f2py.run_main(['-m',out_path,code_path])
    return out_path
    
if __name__=='__main__':
    import os
    wdir = r'C:\Users\aweis\git\pycom\beamforming\fortran'
    fname = 'beamforming_serial.f90'
    file_path = os.path.join(wdir,fname)
    compile_fortran_code(file_path)
    