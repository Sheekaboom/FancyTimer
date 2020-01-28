
Python MATLAB Complex Math Comparisons
========================================
Comparison of MATLAB and python libraries were performed on both the GPU and CPU.
For the CPU MATLAB was compared to Python's Numpy/Scipy and then compared to acceleration with Numba.
For the GPU MATLAB's gpuArray was used and the Python CuPy library was used. Most tests
were performed with both single and double precision complex numbers.

Comparison Results 
-----------------------

These tests were performed on the bb306-awg-1 server. (add info on this system)

.. toctree::
   basic_testing
   extended_testing
   matrix_testing
   beamforming
   fdfd
   timing_code
   :maxdepth: 2
   :caption: Contents:

Running the Comparisons
--------------------------

All comparisons were run together with the following scripts:

Python
++++++++++++

.. module:: pycom.timing.scripts.cpu.time_sweep_cpu
.. data:: pycom.timing.scripts.cpu.time_sweep_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_cpu.py
    :language: python
    :linenos:

MATLAB 
+++++++++++++

.. mat:module:: pycom.timing.scripts.cpu.time_sweep_cpu
.. mat:script:: pycom.timing.scripts.cpu.time_sweep_cpu 

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_cpu.m
    :language: matlab
    :linenos:


Plotting the Data
--------------------------

The output data was plotted with ???



