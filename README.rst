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

.. image:: https://latex.codecogs.com/svg.latex?%5Cmathbf%7BR%7D%3D%5Cmathbf%7BA%7D%5Cmathbf%7BB%7D%5Cmathbf%7BC%7D
    :align: center
    :alt: \mathbf{R}=\mathbf{A}\mathbf{B}\mathbf{C}

can be parenthesized into

.. image:: https://latex.codecogs.com/svg.latex?%5Cmathbf%7BR%7D%3D%5Cleft%28%5Cmathbf%7BA%7D%5Cmathbf%7BB%7D%5Cright%29%5Cmathbf%7BC%7D
    :align: center
    :alt: \mathbf{R}=\left(\mathbf{A}\mathbf{B}\right)\mathbf{C}

or

.. image:: https://latex.codecogs.com/svg.latex?%5Cmathbf%7BR%7D%3D%5Cmathbf%7BA%7D%5Cleft%28%5Cmathbf%7BB%7D%5Cmathbf%7BC%7D%5Cright%29
    :align: center
    :alt: \mathbf{R}=\mathbf{A}\left(\mathbf{B}\mathbf{C}\right)

depending on which one of them incurs less arithmetic cost for the given shapes
of the matrices.  With just a small overhead relative to specialized dynamic
programming code for matrix chain products, general tensor contractions are
supported.  For instance, the ladder term in the CCD theory in quantum chemistry

.. image:: https://latex.codecogs.com/svg.latex?r_%7Babij%7D%3D%5Csum_%7Bc%2Cd%3D1%7D%5Ev%5Csum_%7Bk%2Cl%3D1%7D%5Eov_%7Bklcd%7Dt_%7Bcdij%7Dt_%7Babkl%7D
    :align: center
    :alt: r_{abij}=\sum_{c,d=1}^v\sum_{k,l=1}^ov_{klcd}t_{cdij}t_{abkl}

can be automatically parenthesized into a two-step evaluation

.. image:: https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7Dp_%7Bklij%7D%26%3D%5Csum_%7Bc%2Cd%3D1%7D%5Evv_%7Bklcd%7Dt_%7Bcdij%7D%5C%5Cr_%7Babij%7D%26%3D%5Csum_%7Bk%2Cl%3D1%7D%5Eop_%7Bklij%7Dt_%7Babkl%7D%5Cend%7Baligned%7D
    :align: center
    :alt: \begin{aligned}p_{klij}&=\sum_{c,d=1}^vv_{klcd}t_{cdij}\\r_{abij}&=\sum_{k,l=1}^op_{klij}t_{abkl}\end{aligned}

Because of the efficiency of the algorithm, contraction of even twenty factors
can be handled well.


When computing sums of multiple contractions, factorizations of some or all
terms leading to savings of arithmetic cost can also be automatically found.
For instance, the correlation energy of the CCSD theory in quantum chemistry,

.. image:: https://latex.codecogs.com/svg.latex?e%3D%5Cfrac%7B1%7D%7B4%7D%5Csum_%7Bi%3D1%7D%5E%7Bo%7D%5Csum_%7Bj%3D1%7D%5E%7Bo%7D%5Csum_%7Ba%3D1%7D%5E%7Bv%7D%5Csum_%7Bb%3D1%7D%5E%7Bv%7Du_%7Bijab%7D%5C%2Ct%5E%7B%282%29%7D_%7Babij%7D%2B%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bo%7D%5Csum_%7Bj%3D1%7D%5E%7Bo%7D%5Csum_%7Ba%3D1%7D%5E%7Bv%7D%5Csum_%7Bb%3D1%7D%5E%7Bv%7Du_%7Bijab%7D%5C%2Ct%5E%7B%281%29%7D_%7Bai%7D%5C%2Ct%5E%7B%281%29%7D_%7Bbj%7D
    :align: center
    :alt: e=\frac{1}{4}\sum_{i=1}^{o}\sum_{j=1}^{o}\sum_{a=1}^{v}\sum_{b=1}^{v}u_{ijab}\,t^{(2)}_{abij}+\frac{1}{2}\sum_{i=1}^{o}\sum_{j=1}^{o}\sum_{a=1}^{v}\sum_{b=1}^{v}u_{ijab}\,t^{(1)}_{ai}\,t^{(1)}_{bj}

can be automatically rewritten into

.. image:: https://latex.codecogs.com/svg.latex?e%3D%5Cfrac%7B1%7D%7B4%7D%5Csum_%7Bi%3D1%7D%5E%7Bo%7D%5Csum_%7Bj%3D1%7D%5E%7Bo%7D%5Csum_%7Ba%3D1%7D%5E%7Bv%7D%5Csum_%7Bb%3D1%7D%5E%7Bv%7Du_%7Bijab%7D%5Cleft%28t%5E%7B%282%29%7D_%7Babij%7D%2B2t%5E%7B%281%29%7D_%7Bai%7Dt%5E%7B%281%29%7D_%7Bbj%7D%5Cright%29
    :align: center
    :alt: e=\frac{1}{4}\sum_{i=1}^{o}\sum_{j=1}^{o}\sum_{a=1}^{v}\sum_{b=1}^{v}u_{ijab}\left(t^{(2)}_{abij}+2t^{(1)}_{ai}t^{(1)}_{bj}\right)

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

