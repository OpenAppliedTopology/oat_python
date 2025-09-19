


Install
===================

**Python users** Can install oat_python through
`PyPI <https://pypi.org/project/oat_python/>`__ using any package
manager, for example ``pip install oat_python``,
``conda_install oat_python``, etc. 

**Developers** Everyone is a developer, and everyone is invited to
modify and extend the source code for this package! oat_python is a
combination of Rust and Python code. We developed oat_python using
``PyO3`` and ``maturin``. To download and modify oat_python, then
install and use the modified version, check out the 
`github repository <https://github.com/OpenAppliedTopology/oat_python>`__.




API
=============
  


Homology
--------------

.. automodule:: oat_python.homology
   :members:
   :undoc-members:
   :show-inheritance: 



.. _vietoris-rips-section:

Vietoris Rips 
--------------


The :mod:`oat_python.core.vietoris_rips` module provides tools to work with :term:`Vietoris-Rips complexes <vietoris rips complex>`.

It can be used to

- compute persistent homology
- access rows and columns of a boundary matrix
- obtain the (co)boundary of a linear combination of simplices

and more!

.. automodule:: oat_python.core.vietoris_rips
   :members:
   :undoc-members:
   :show-inheritance:    


Point clouds
-----------------   

.. automodule:: oat_python.point_cloud
   :members:
   :undoc-members:
   :show-inheritance:      


.. _hypergraph-section:


Hypergraphs
-----------------

.. automodule:: oat_python.core.dowker
   :members:
   :undoc-members:
   :show-inheritance:

Additional tools
~~~~~~~~~~~~~~~~~~~
.. automodule:: oat_python.hypergraph
   :members:
   :undoc-members:
   :show-inheritance:


Plotting
-----------------   

.. automodule:: oat_python.plot
   :members:
   :undoc-members:
   :show-inheritance:

Simplices
-----------------   

.. automodule:: oat_python.simplex
   :members:
   :undoc-members:
   :show-inheritance:

Matrices 
--------------

.. automodule:: oat_python.matrix
   :members:
   :undoc-members:
   :show-inheritance:   

Dissimilarity 
--------------

.. automodule:: oat_python.dissimilarity
   :members:
   :undoc-members:
   :show-inheritance:   


Barcodes 
--------------

.. automodule:: oat_python.barcode
   :members:
   :undoc-members:
   :show-inheritance:    



Core
--------------

.. automodule:: oat_python.core
   :members:
   :undoc-members:
   :show-inheritance:    


