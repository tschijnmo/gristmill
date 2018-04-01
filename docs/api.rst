API Reference
-------------

The ``gristmill`` package can be divided into two orthogonal parts,

The evaluation optimization part,
    which transforms tensor definitions into a mathematically equivalent
    definition sequence with less floating-point operations required.

The code generation part,
    which takes tensor definitions, either optimized or not, into computer code
    snippets.


.. py:currentmodule:: gristmill


Evaluation Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: optimize

.. autoclass:: ContrStrat
    :members:

.. autoclass:: RepeatedTermsStrat
    :members:

.. autofunction:: verify_eval_seq

.. autofunction:: get_flop_cost


Code generation
~~~~~~~~~~~~~~~

Code generation system
......................

.. autoclass:: BasePrinter
    :members:
    :special-members:

.. autofunction:: mangle_base

.. autoclass:: NaiveCodePrinter
    :members:
    :special-members:

.. autoclass:: CCodePrinter
    :members:
    :special-members:

.. autoclass:: FortranPrinter
    :members:
    :special-members:

.. autoclass:: EinsumPrinter
    :members:
    :special-members:


Internal facilities for printer writers
.......................................

.. autoclass:: gristmill.utils.JinjaEnv
    :members:
    :special-members:


.. autoclass:: gristmill.generate.TensorDecl

.. autoclass:: gristmill.generate.BeginBody

.. autoclass:: gristmill.generate.BeforeComp

.. autoclass:: gristmill.generate.CompTerm

.. autoclass:: gristmill.generate.OutOfUse

.. autoclass:: gristmill.generate.EndBody

