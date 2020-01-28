
Matrix Operations
=====================
This covers decomposition, multiplication, and solving of linear systems.
- Matrix Multiplication : :math:`C=A*B`
- LU Decomposition : :math:`C=lu(A)`
- Dense Matrix Solve : :math:`A\backslash b`
- Sparse Matrix Solve : :math:`A\backslash b`

Matrix Multiplication
-------------------------
.. math:: C=A*B

CPU
+++++++++
.. raw:: html
	:file: figs/matmul.html

LU Decomposition
--------------------
.. math:: C=lu(A)

CPU
+++++++++
.. raw:: html
	:file: figs/lu.html

Dense Matrix Solving
-----------------------
.. math:: A\backslash b

CPU
++++++++++++
.. raw:: html
	:file: figs/solve.html

Sparse Matrix Solving
-----------------------
.. math:: A\backslash b

CPU
++++++++++++
.. raw:: html
	:file: figs/ssolve.html
    
Code
---------
This was generated with the following scripts:

Python
+++++++++

**CPU Code**

..  module:: pycom.timing.scripts.cpu.time_sweep_matrix_cpu
..  data:: pycom.timing.scripts.cpu.time_sweep_matrix_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_matrix_cpu.py
    :language: python
    :linenos:

..  module:: pycom.timing.scripts.cpu.time_sweep_sparse_cpu
..  data:: pycom.timing.scripts.cpu.time_sweep_sparse_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_sparse_cpu.py 
    :language: python
    :linenos:
        
MATLAB
++++++++++

**CPU Code**

.. mat:module:: pycom.timing.scripts.cpu.time_sweep_matrix_cpu
.. mat:script:: pycom.timing.scripts.cpu.time_sweep_matrix_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_matrix_cpu.m 
    :language: matlab
    :linenos:

.. mat:module:: pycom.timing.scripts.cpu.time_sweep_sparse_cpu
.. mat:script:: pycom.timing.scripts.cpu.time_sweep_sparse_cpu

.. literalinclude:: /../pycom/timing/scripts/cpu/time_sweep_sparse_cpu.m 
    :language: matlab
    :linenos:
        
        





