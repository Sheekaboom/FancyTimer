
Python MATLAB Complex Math Comparisons
========================================
Comparison of MATLAB and python libraries were performed on both the GPU and CPU.
For the CPU MATLAB was compared to Python's Numpy/Scipy and then compared to acceleration with Numba.
For the GPU MATLAB's gpuArray was used and the Python CuPy library was used.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Basic Operations
-------------------
add/sub/mult/divide

CPU
++++++++
results

GPU
++++++
results

Matrix Operations
-------------------
matmul/lu decomp

CPU
++++++++
results

GPU
++++++
results

Matrix Solving
-------------------
Dense/Sparse

CPU
++++++++
results

GPU
++++++
results

Fast Fourier Transform (FFT)
------------------------------
??? length fft

CPU
++++++++
results

GPU
++++++
results

   
Python MATLAB Beamforming Comparisons 
========================================
This was tested for ???


CPU
+++++++++
results

GPU
+++++++++
results
	
Finite Difference Frequency Domain (FDFD) Simulation
======================================================
Run times between MATLAB and python were tested for a finite difference frequency domain (FDFD) simulation for scattering from a cylinder.

.. raw:: html
	:file: figs/FDFD_results.html
	
	
FDFD Code
-----------
.. automodule:: python_matlab_comparison.FDFD.FDFD_2D
	:members:


