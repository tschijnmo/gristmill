.. image:: https://travis-ci.org/tschijnmo/gristmill.svg?branch=master
    :target: https://travis-ci.org/tschijnmo/gristmill


gristmill
~~~~~~~~~


Gristmill is a package based on the `drudge`_ algebra system for automatic
optimization and code generation of tensor computations.  In spite of being
designed for applications in quantum chemistry and many-body theory, gristmill
can be used for any scientific computing problem with dependency on tensor
computations.


The optimizer utilizes novel advanced algorithm to efficiently parenthesize and
factorize tensor computations for less arithmetic cost.  For instance, a matrix
chain product

.. math::

    \mathbf{R} = \mathbf{A} \mathbf{B} \mathbf{C}

can be parenthesized into

.. math::

    \mathbf{R} = \left( \mathbf{A} \mathbf{B} \right) \mathbf{C}

or

.. math::

    \mathbf{R} = \mathbf{A} \left( \mathbf{B} \mathbf{C} \right)

depending on which one of them incurs less arithmetic cost for the given shapes
of the matrices.  With just a small overhead relative to specialized dynamic
programming code for matrix chain products, general tensor contractions are
supported.  For instance, the ladder term in the CCD theory in quantum chemistry

.. math::

    r_{abij} = \sum_{c,d=1}^v \sum_{k,l=1}^o v_{klcd} t_{cdij} t_{abkl}

can be automatically parenthesized into a two-step evaluation

.. math::

    \begin{aligned}
        p_{klij} &= \sum_{c,d=1}^v v_{klcd} t_{cdij}\\
        r_{abij} &= \sum_{k,l=1}^o p_{klij} t_{abkl}\\
    \end{aligned}

Because of the efficiency of the algorithm, contraction of even twenty factors
can be handled well.


When computing sums of multiple contractions, factorizations of some or all
terms leading to savings of arithmetic cost can also be automatically found.
For instance, the correlation energy of the CCSD theory in quantum chemistry,

.. math::

    e = \frac{1}{4} \sum_{i,j=1}^o \sum_{a,b=1}^{v} u_{ijab} t^{(2)}_{abij}
    + \frac{1}{2} \sum_{i,j=1}^o \sum_{a,b=1}^v u_{ijab} t^{(1)}_{ai} t^{(1)}_{bj}

can be automatically rewritten into

.. math::

    e = \frac{1}{4} \sum_{i,j=1}^o \sum_{a,v=1}^v u_{ijab} \left(
        t^{(2)}_{abij} + 2 t^{(1)}_{ai} t^{(1)}_{bj}
    \right)

which takes less arithmetic cost.

In addition to parenthesization and factorization, gristmill also has additional
optimization heuristics like common symmetrization optimization.  The same
intermediates can also be guaranteed to be computed only once by the
canonicalization power of `drudge`_.


The code generator is a component orthogonal to the optimizer.  Both optimized
and unoptimized computation can be given for naive Fortran or C code (with
optional OpenMP parallelization), or Python code using NumPy or TensorFlow
libraries.


Gristmill is developed by Jinmo Zhao and Prof Gustavo E Scuseria at Rice
University, and was supported as part of the Center for the Computational Design
of Functional Layered Materials, an Energy Frontier Research Center funded by
the U.S. Department of Energy, Office of Science, Basic Energy Sciences under
Award DE-SC0012575.


.. _drudge: https://github.com/tschijnmo/drudge

