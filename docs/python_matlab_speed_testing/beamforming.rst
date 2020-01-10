Beamforming Simulation
========================================

Beamforming was tested with a synthetic 35x35 element array at 40GHz each element was spaced
int he x and y directions by :math:`\frac{\lambda}{2}` calculated at 40 GHz. For the speed
results, the number of angles calculated were swept from 1 to 500 angles in both the azimuthal
and elevation dimensions. These different angles were then used to create a meshgrid providing
a total number of calculated angles as the input to :func:`pycom.python_matlab_comparison.beamforming.beamform_speed.beamform_speed`
as num_angles^2. All tests were run with a single incident plane wave simulated at [pi/4,-pi/4].

Output results
------------------
181 angles (1 degree steps) in both azimuth and elevation provides the following 3D output.
This is done with a single incident plane wave simulated at [pi/4,-pi/4].


.. raw:: html
	:file: figs/beamform_results.html


Speed Comparison
--------------------
Speed comparisons between python and MATLAB can be seen below:
    
CPU
+++++++++

.. raw:: html
    :file: figs/beamforming.html


GPU
+++++++++
results


Code 
-------

The following module was used to calculate the Beamforming solution:

.. automodule:: python_matlab_comparison.beamforming.beamform_speed
	:members:
    
The runtimes were generated with the following scripts:

Python
+++++++++

**CPU Code** 

..  module:: python_matlab_comparison.scripts.cpu.time_sweep_beamforming_cpu
..  data:: python_matlab_comparison.scripts.cpu.time_sweep_beamforming_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_beamforming_cpu.m 
    :language: python
    :linenos:



MATLAB
+++++++++

**CPU Code**

.. mat:module:: python_matlab_comparison.scripts.cpu.time_sweep_beamforming_cpu
.. mat:script:: python_matlab_comparison.scripts.cpu.time_sweep_beamforming_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_beamforming_cpu.m 
    :language: matlab
    :linenos:
	



