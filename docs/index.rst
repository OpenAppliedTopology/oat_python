.. oat_python documentation master file, created by
   sphinx-quickstart on Thu May 22 21:20:58 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/banner.png
   :width: 80%
   :align: center
   :alt: Project Banner

Welcome
==============================================

Welcome to the user guide for the `oat_python` package.  `Open applied topology (OAT) <https://openappliedtopology.github.io>`__
is a library for fast, user-friendly algebra and topology. OAT has

- a user-friendly frontend for Python users, called
  `oat_python <https://github.com/OpenAppliedTopology/oat_python>`__
- a fast backend written in Rust, called
  `oat_rust <https://github.com/OpenAppliedTopology/oat_rust>`__
  
This page contains the user guide for
`oat_python <https://github.com/OpenAppliedTopology/oat_python>`__.

Caution: breaking changes
=========================

OAT is in early stages of develpoment, and it’s evolving quickly. Code
that you write today may not work tomorrow, due to these changes. We
will do our very best to make sure that if/when this happens, you will
only need to make small changes to your code to fix the problem (e.g.,
updating the name of a function). However, please do bear this in mind
as you write your code!


Community
===========================================

OAT is by and for the open source community.  Reach out to the developers if you

- Need help getting started
- Wish for a missing feature
- Want to try coding
  
A collaboration of 20 research centers at colleges, universities, private, and public organizations support OAT's
development. The founding developers are Princton University, Macalester College, and the University of Delaware
The National Science Foundation granted seed funding for OAT in
2019 under the `ExHACT <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1854748&HistoricalAwards=false>`_
project, and `Pacific Northwest National Laboratory <https://www.pnnl.gov/>`_ (PNNL) provides continuing financial
support.  PNNL now coordinates development.  See `here <https://github.com/OpenAppliedTopology/oat_python/blob/main/ATTRIBUTIONS.md>`_ for further details.

Values
===========================================

Our `shared values <https://github.com/OpenAppliedTopology/oat_python/blob/main/CODE_OF_CONDUCT.md>`_ are

- Inclusion
- Respect, and 
- A shared passion to expand human knowledge, through algebraic topology


Mission
===========================================

**Performance**
    
OAT is a first-class solver for cutting-edge applications.  It is ideally suited to large, sparse data sets.
The core library is written in Rust, a low-level systems programming language designed for safety and performance.
High-level wrappers are available in Python. 


**Reliability**

OAT has more unit tests than type definitions and function definitions, combined.
Its modular design enables end users to write their own checks for correctness, with ease.
The library inherits strong safety guarantees from the the Rust compiler.


**Transparency**

OAT documentation emphasizes clarity and accessibility for users with all backgrounds.  It includes more than 180 working examples, and describes both code and underlying mathematical concepts in detail. 
The :ref:`tutorials` illustrate how to combine multiple tools into larger applications.
The platform's modular design breaks large solvers into basic components, which it exposes to the user for inspection.  In addition, the library provides powerful methods to inspect and analyze objects, consistent with the way humans naturally think about problems; for example, you can look up rows and columns of boundary matrices using *cubes*, *simplices*, or *cells* as keys.
  

**Modularity**

OAT reduces complex problems to the same basic building blocks that topologists use when writing on a chalk board. Users can mix and match those blocks with confidence, using a simple, streamlined interface.  They can even create new components that work seemlessly with the rest of the library, including coefficient rings, sparse matrix data structures, and customized filtrations on simplicial complexes.





Contributing and understanding the code
===========================================

For information on **contributing** including an overview of how the code in this project
is organized, see
`CONTRIBUTING.md <https://github.com/OpenAppliedTopology/oat_python/blob/main/CONTRIBUTING.md>`__.

License
=======

For inforamtion on copyright and licensing, see
`LICENSE <https://github.com/OpenAppliedTopology/oat_python/blob/main/LICENSE>`__.

Attributions
============

OAT is an extension of the ExHACT library. See
`ATTRIBUTIONS.md <https://github.com/OpenAppliedTopology/oat_python/blob/main/ATTRIBUTIONS.md>`__
for details.



Table of Contents
=================

.. toctree::
   :maxdepth: 3

   sidebar
   glossary
   auto_examples/index
   faq
  
..  auto_examples
..  auto_examples/index   



