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

.. image:: http://www.sciweavers.org/tex2img.php?eq=%5Cmathbf%7BR%7D%20%3D%20%5Cmathbf%7BA%7D%20%5Cmathbf%7BB%7D%20%5Cmathbf%7BC%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: \mathbf{R} = \mathbf{A} \mathbf{B} \mathbf{C}
    :width: 75
    :height: 14

can be parenthesized into

.. image:: http://www.sciweavers.org/tex2img.php?eq=%5Cmathbf%7BR%7D%20%3D%20%5Cleft%28%5Cmathbf%7BA%7D%20%5Cmathbf%7BB%7D%5Cright%29%20%5Cmathbf%7BC%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: \mathbf{R} = \left(\mathbf{A} \mathbf{B}\right) \mathbf{C}
    :width: 92
    :height: 19

or

.. image:: http://www.sciweavers.org/tex2img.php?eq=%5Cmathbf%7BR%7D%20%3D%20%5Cmathbf%7BA%7D%20%5Cleft%28%20%5Cmathbf%7BB%7D%20%5Cmathbf%7BC%7D%5Cright%29&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: \mathbf{R} = \mathbf{A} \left( \mathbf{B} \mathbf{C}\right)
    :width: 92
    :height: 19

depending on which one of them incurs less arithmetic cost for the given shapes
of the matrices.  With just a small overhead relative to specialized dynamic
programming code for matrix chain products, general tensor contractions are
supported.  For instance, the ladder term in the CCD theory in quantum chemistry

.. image:: http://www.sciweavers.org/tex2img.php?eq=%20%20r_%7Ba%2C%20b%2C%20i%2C%20j%7D%20%3D%20%5Csum_%7Bc%2C%20d%20%3D%201%7D%5Ev%20%5Csum_%7Bk%2C%20l%20%3D%201%7D%5Eo%0A%20%20v_%7Bk%2C%20l%2C%20c%2C%20d%7D%20t_%7Bc%2C%20d%2C%20i%2C%20j%7D%20t_%7Ba%2C%20b%2C%20k%2C%20l%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: r_{a, b, i, j} = \sum_{c, d = 1}^v \sum_{k, l = 1}^o  v_{k, l, c, d} t_{c, d, i, j} t_{a, b, k, l}
    :width: 244
    :height: 47

can be automatically parenthesized into a two-step evaluation

.. image:: http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Baligned%7D%0A%20%20p_%7Bk%2C%20l%2C%20i%2C%20j%7D%20%26%3D%20%5Csum_%7Bc%2C%20d%20%3D%201%7D%5Ev%20v_%7Bk%2C%20l%2C%20c%2C%20d%7D%20t_%7Bc%2C%20d%2C%20i%2C%20j%7D%20%5C%5C%0A%20%20r_%7Ba%2C%20b%2C%20i%2C%20j%7D%20%26%3D%20%5Csum_%7Bk%2C%20l%20%3D%201%7D%5Eo%20p_%7Bk%2C%20l%2C%20i%2C%20j%7D%20t_%7Ba%2C%20b%2C%20k%2C%20l%7D%20%5C%5C%0A%5Cend%7Baligned%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: \begin{aligned}  p_{k, l, i, j} &= \sum_{c, d = 1}^v v_{k, l, c, d} t_{c, d, i, j} \\  r_{a, b, i, j} &= \sum_{k, l = 1}^o p_{k, l, i, j} t_{a, b, k, l} \end{aligned}
    :width: 175
    :height: 100

Because of the efficiency of the algorithm, contraction of even twenty factors
can be handled well.


When computing sums of multiple contractions, factorizations of some or all
terms leading to savings of arithmetic cost can also be automatically found.
For instance, the correlation energy of the CCSD theory in quantum chemistry,

.. image:: http://www.sciweavers.org/tex2img.php?eq=%20%20%20%20%20%20e%20%3D%0A%20%20%20%20%20%20%5Cfrac%7B1%7D%7B4%7D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bo%7D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bo%7D%0A%20%20%20%20%20%20%5Csum_%7Ba%20%3D%201%7D%5E%7Bv%7D%20%5Csum_%7Bb%20%3D%201%7D%5E%7Bv%7D%20u_%7Bijab%7D%20%5C%2C%20t%5E%7B%282%29%7D_%7Babij%7D%0A%20%20%20%20%20%20%2B%0A%20%20%20%20%20%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bo%7D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bo%7D%0A%20%20%20%20%20%20%5Csum_%7Ba%20%3D%201%7D%5E%7Bv%7D%20%5Csum_%7Bb%20%3D%201%7D%5E%7Bv%7D%20u_%7Bijab%7D%20%5C%2C%20t%5E%7B%281%29%7D_%7Bai%7D%20%5C%2C%20t%5E%7B%281%29%7D_%7Bbj%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: e = \frac{1}{4} \sum_{i = 1}^{o} \sum_{j = 1}^{o} \sum_{a = 1}^{v} \sum_{b = 1}^{v} u_{ijab} \, t^{(2)}_{abij} + \frac{1}{2} \sum_{i = 1}^{o} \sum_{j = 1}^{o} \sum_{a = 1}^{v} \sum_{b = 1}^{v} u_{ijab} \, t^{(1)}_{ai} \, t^{(1)}_{bj}
    :width: 419
    :height: 49

can be automatically rewritten into

.. image:: http://www.sciweavers.org/tex2img.php?eq=e%20%3D%0A%20%20%20%20%20%20%5Cfrac%7B1%7D%7B4%7D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bo%7D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bo%7D%0A%20%20%20%20%20%20%5Csum_%7Ba%20%3D%201%7D%5E%7Bv%7D%20%5Csum_%7Bb%20%3D%201%7D%5E%7Bv%7D%20u_%7Bijab%7D%0A%20%20%20%20%20%20%5Cleft%28%20t%5E%7B%282%29%7D_%7Babij%7D%20%2B%202%20t%5E%7B%281%29%7D_%7Bai%7D%20t%5E%7B%281%29%7D_%7Bbj%7D%20%5Cright%29&bc=Transparent&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
    :align: center
    :alt: e = \frac{1}{4} \sum_{i = 1}^{o} \sum_{j = 1}^{o} \sum_{a = 1}^{v} \sum_{b = 1}^{v} u_{ijab} \left( t^{(2)}_{abij} + 2 t^{(1)}_{ai} t^{(1)}_{bj} \right)
    :width: 301
    :height: 49

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

