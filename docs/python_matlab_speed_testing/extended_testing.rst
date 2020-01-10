
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

..  module:: python_matlab_comparison.scripts.cpu.time_sweep_extended_cpu
..  data:: python_matlab_comparison.scripts.cpu.time_sweep_extended_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_extended_cpu.py
    :language: python
    :linenos:
        

MATLAB
+++++++++

**CPU Code**

.. mat:module:: python_matlab_comparison.scripts.cpu.time_sweep_extended_cpu
.. mat:script:: python_matlab_comparison.scripts.cpu.time_sweep_extended_cpu

.. literalinclude:: /../python_matlab_comparison/scripts/cpu/time_sweep_extended_cpu.m 
    :language: matlab
    :linenos:
    
    
    
	
	
	
        
        
        





