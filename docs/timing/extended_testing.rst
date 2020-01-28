
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

..  module:: pycom.timing.scripts.cpu.time_sweep_extended_cpu
..  data:: pycom.timing.scripts.cpu.time_sweep_extended_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_extended_cpu.py
    :language: python
    :linenos:
        

MATLAB
+++++++++

**CPU Code**

.. mat:module:: pycom.timing.scripts.cpu.time_sweep_extended_cpu
.. mat:script:: pycom.timing.scripts.cpu.time_sweep_extended_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_extended_cpu.m 
    :language: matlab
    :linenos:
    
    
    
	
	
	
        
        
        





