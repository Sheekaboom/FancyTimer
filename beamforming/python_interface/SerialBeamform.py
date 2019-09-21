'''
@author ajw
@date 8-24-2019
@brief functions and classes for creating
    python functions from fortran subroutines
'''
import numpy as np
import os

from SpeedBeamform import SpeedBeamform
from SpeedBeamform import get_ctypes_pointers
import ctypes
wdir = os.path.dirname(os.path.realpath(__file__))


serial_fortran_library_path = os.path.join(wdir,'../fortran/build/libbeamform_serial.dll')
class SerialBeamformFortran(SpeedBeamform):
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    def __init__(self):
        '''
        @brief constructor
        '''
        super().__init__(serial_fortran_library_path)
        
    def get_steering_vectors(self,frequency,positions,az,el):
        '''
        @brief get steering vectors with inputs same as super().get_steering_vectors()
        @note all angles are in degrees
        '''
        az = np.deg2rad(az,dtype=self.precision_types['float'])
        el = np.deg2rad(el,dtype=self.precision_types['float'])
        positions = np.array(positions,dtype=self.precision_types['float'])
        nazel = len(az) #number of azimuth elevations
        npos = positions.shape[0] #number of positions
        steering_vectors = np.zeros((nazel,npos),dtype=self.precision_types['complex'])
        #now setup the ctypes stuff
        cfreq = ctypes.pointer(self.precision_types['ctfloat'](frequency))
        cpos = positions.ctypes; caz = az.ctypes; cel = el.ctypes
        csv = steering_vectors.ctypes
        cnazel = ctypes.pointer(ctypes.c_int(nazel))
        cnp = ctypes.pointer(ctypes.c_int(npos))
        self._lib.get_steering_vectors(cfreq,cpos,caz,cel,csv,cnp,cnazel)
        return steering_vectors
    
    def get_beamformed_values(self,freqs,positions,weights,meas_vals,az,el):
        '''
        @brief get steering vectors with inputs same as super().get_steering_vectors()
        @note all angles are in degrees
        '''
        #perform input checks
        meas_vals
        #now pass the values
        az = np.deg2rad(az,dtype=self.precision_types['float'])
        el = np.deg2rad(el,dtype=self.precision_types['float'])
        freqs = np.array(freqs,dtype=self.precision_types['float'])
        positions = np.array(positions,dtype=self.precision_types['float'])
        weights = np.array(weights,dtype=self.precision_types['complex'])
        meas_vals = np.array(meas_vals,dtype=self.precision_types['complex'])
        nazel = az.size
        nfreqs = freqs.size
        npos = positions.shape[0] #number of positions
        out_vals = np.zeros((nfreqs,nazel),dtype=self.precision_types['complex'])
        #now setup the ctypes stuff
        cfreqs = freqs.ctypes; cweights = weights.ctypes
        cmeas = meas_vals.ctypes
        cpos = positions.ctypes; caz = az.ctypes; cel = el.ctypes
        cov = out_vals.ctypes
        cnf = ctypes.pointer(ctypes.c_int(nfreqs))
        cnazel = ctypes.pointer(ctypes.c_int(nazel))
        cnp = ctypes.pointer(ctypes.c_int(npos))
        self._lib.get_beamformed_values(cfreqs,cpos,cweights,cmeas,caz,cel,cov,cnf,cnp,cnazel)
        return out_vals
        
        
class SerialBeamformPython(SpeedBeamform):
    '''
    @brief superclass for BEAMforming FORTran code
    '''
    def __init__(self):
        '''
        @brief constructor
        '''
        super().__init__(serial_fortran_library_path)
        
    def get_k(freq,eps_r=1,mu_r=1):
        '''
        @brief get our wavenumber
        '''
        sol = 299792458.0
        lam = sol/np.sqrt(eps_r*mu_r)/freq
        k = 2*np.pi/lam
        return k                

if __name__=='__main__':
    n = 2
    cnp = ctypes.pointer(ctypes.c_int(n))
    mysbf = SerialBeamformFortran()
    
    #test get steering vectors
    freqs = [40e9] #frequency
    #freqs = np.arange(26.5e9,40e9,10e6)
    spacing = 2.99e8/np.max(freqs)/2 #get our lambda/2
    numel = [35,35,1] #number of elements in x,y
    Xel,Yel,Zel = np.meshgrid(np.arange(numel[0])*spacing,np.arange(numel[1])*spacing,np.arange(numel[2])*spacing) #create our positions
    pos = np.stack((Xel.flatten(),Yel.flatten(),Zel.flatten()),axis=1) #get our position [x,y,z] list
    #az = np.arange(-90,90,1)
    #az = np.array([-90,45,0,45 ,90])
    az = [-45]
    el = np.zeros_like(az)
    sv = mysbf.get_steering_vectors(freqs[0],pos,az,el)
    meas_vals = np.tile(sv[0],(len(freqs),1)) #syntethic plane wave
    #meas_vals = np.ones((1,35))
    #print(np.rad2deg(np.angle(sv)))
    #print(sv)
    #print(np.real(sv))
    azl = np.arange(-90,90,1)
    ell = np.arange(-90,90,1)
    AZ,EL = np.meshgrid(azl,ell)
    az = AZ.flatten()
    el = EL.flatten()
    #az = np.array([-90,45,0,45,90])
    #az = [-90]
    el = np.zeros_like(az)
    az = azl
    bf_vals = mysbf.get_beamformed_values(freqs,pos,np.ones((pos.shape[0])),meas_vals,az,el)
    #print(bf_vals)
    
    def get_k_vec(freq,az,el):
        sol = 299792458.0
        lam = sol/freq
        k = 2*np.pi/lam
        az = np.deg2rad(az)
        el = np.deg2rad(el)
        kv = k*np.array([np.sin(az)*np.cos(el),np.sin(el),np.cos(az)*np.cos(el)]).transpose()
        return kv

    #print(get_k_vec(freq,az,el))
        
        
    import matplotlib.pyplot as plt
    plt.plot(az,10*np.log10(np.abs(bf_vals[0])))
    '''
    #test array multiply
    arr1 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arr2 = (np.random.rand(n)+1j*np.random.rand(n)).astype(np.csingle)
    arro = np.zeros_like(arr1)
    mysbf.test_complex_array_multiply(arr1.ctypes,arr2.ctypes,arro.ctypes,cnp)
    nparro = arr1*arr2
    print(all(nparro==arro))
    
    #test matrix multiply
    mat1 = (np.random.rand(n,n)+1j*np.random.rand(n,n)).astype(np.csingle)
    mat2 = (np.random.rand(n,n)+1j*np.random.rand(n,n)).astype(np.csingle)
    mato = np.zeros_like(mat1)
    mysbf.test_complex_matrix_multiply(mat1.ctypes,mat2.ctypes,mato.ctypes,cnp)
    npmato = mat1*mat2
    
    #test 2d sum
    sarro = np.zeros_like(arr1)
    mysbf.test_complex_sum(mat1.ctypes,sarro.ctypes,cnp)
    npsarro = np.sum(mat1,axis=0)
    print(all(npsarro==sarro))
    '''    
        
        
        
        
        
    
    
    