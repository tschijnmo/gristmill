"""
Test of the single-term optimization based on matrix chain product.

"""

import pytest
from drudge import Range, Drudge, TensorDef
from sympy import symbols, IndexedBase

from gristmill import optimize


def test_matrix_chain(spark_ctx):
    """Test a basic matrix chain multiplication problem.

    Matrix chain multiplication problem is the classical problem that motivated
    the algorithm for single-term optimization in this package.  So here a very
    simple matrix chain multiplication problem with three matrices are used to
    test the factorization facilities.  In this simple test, we will have three
    matrices :math:`x`, :math:`y`, and :math:`z`, which are of shapes
    :math:`m\\times n`, :math:`n \\times l`, and :math:`l \\times m`
    respectively. In the factorization, we are going to set :math:`n = 2 m` and
    :math:`l = 3 m`.

    If we multiply the first two matrices first, the cost will be (two times)

    .. math::

        m n l + m^2 l

    Or if we multiply the last two matrices first, the cost will be (two times)

    .. math::

        m n l + m^2 n

    In addition to the classical matrix chain product, also tested is the
    trace of their cyclic product.

    .. math::

        t = \\sum_i \\sum_j \\sum_k x_{i, j} y_{j, k} z_{k, i}

    If we first take the product of :math:`Y Z`, the cost will be (two times)
    :math:`n m l + n m`. For first multiplying :math:`X Y` and :math:`Z X`,
    the costs will be (two times) :math:`n m l + m l` and :math:`n m l + n l`
    respectively.

    """

    #
    # Basic context setting-up.
    #

    dr = Drudge(spark_ctx)

    # The sizes.
    m, n, l = symbols('m n l')

    # The ranges.
    m_range = Range('M', 0, m)
    n_range = Range('N', 0, n)
    l_range = Range('L', 0, l)

    dr.set_dumms(m_range, symbols('a b c'))
    dr.set_dumms(n_range, symbols('i j k'))
    dr.set_dumms(l_range, symbols('p q r'))
    dr.add_resolver_for_dumms()

    # The indexed bases.
    x = IndexedBase('x', shape=(m, n))
    y = IndexedBase('y', shape=(n, l))
    z = IndexedBase('z', shape=(l, m))

    # The costs substitution.
    substs = {
        n: m * 2,
        l: m * 3
    }

    #
    # Actual tests.
    #

    p = dr.names

    target_base = IndexedBase('t')
    target = TensorDef(
        target_base, (p.a, p.M), (p.b, p.M),
        dr.einst(x[p.a, p.i] * y[p.i, p.p] * z[p.p, p.b])
    )

    # Perform the factorization.
    targets = [target]
    eval_seq = optimize(targets, substs=substs)
    assert len(eval_seq) == 2
