Introduction
------------

.. py:currentmodule:: gristmill

For tensor definition objects from drudge, as long as none of their terms has
any vector part, they can be readily given to gristmill for optimization and
(or) code generation.  For optimization, basically an iterable of tensor
definitions can be given to the :py:func:`optimize` function, then a
mathematically-equivalent list of tensor definitions will be returned, possibly
incurring much less arithmetic cost.   For any iterable of tensor definitions,
whether from the gristmill optimizer or not, the code printers
:py:class:`FortranPrinter`, :py:class:`CCodePrinter`, and
:py:class:`EinsumPrinter` can be used to generate code automatically.  The exact
form of the generated code is very tunable.
