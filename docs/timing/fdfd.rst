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

..  module:: python_matlab_comparison.scripts.cpu.time_sweep_fdfd_cpu
..  data:: python_matlab_comparison.scripts.cpu.time_sweep_fdfd_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_fdfd_cpu.py
    :language: python
    :linenos:




MATLAB
++++++++++++

**CPU Code**

.. mat:module:: python_matlab_comparison.scripts.cpu.time_sweep_fdfd_cpu
.. mat:script:: python_matlab_comparison.scripts.cpu.time_sweep_fdfd_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_fdfd_cpu.m 
    :language: matlab
    :linenos:






