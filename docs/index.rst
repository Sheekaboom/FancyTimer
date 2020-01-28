.. PyCom documentation master file, created by
   sphinx-quickstart on Tue Dec 17 21:43:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyCom's documentation!
=================================

This documentation covers the pycom library (maybe it should be renamed Py5G?) 
which contains code to implement and test angle of arrival (AoA) and Beamforming type algorithms (may require some samurai code)

.. toctree::
   modulation/index
   :maxdepth: 2
   :caption: Communications Simulation:

.. toctree::
   aoa/index
   :maxdepth: 2
   :caption: Angle of Arrival:
   
.. toctree::
   timing/index
   :maxdepth: 2
   :caption: Python/MATLAB Speed Comparison and Timing Functions:
   
Dependencies
--------------

- numpy
- samurai (custom NIST SAMURAI system software)
- plolty
- scipy
- What else ???


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* `MATLAB Module Index <mat-modindex.html>`_
* :ref:`search`
