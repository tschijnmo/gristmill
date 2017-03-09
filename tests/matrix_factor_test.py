"""Test of matrix factorization."""

from drudge import Range, Drudge
from sympy import symbols, IndexedBase, Symbol

from gristmill import optimize


def test_matrix_factorization(spark_ctx):
    """Test a basic matrix multiplication factorization problem.

    In this test, there are four matrices involved, X, Y, U, and V.  The final
    expression to optimize is mathematically

    .. math::

        (2 X - Y) * (2 U + V)

    Here, the expression is to be given in its extended form originally, and we
    test if it can be factorized into something similar to what we have above.
    Here we have the signs and coefficients to have better code coverage for
    these cases.

    """

    #
    # Basic context setting-up.
    #

    dr = Drudge(spark_ctx)

    n = Symbol('n')
    r = Range('R', 0, n)

    dumms = symbols('a b c d')
    a, b, c, d = dumms
    dr.set_dumms(r, dumms)
    dr.add_resolver_for_dumms()

    # The indexed bases.
    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    # The target.
    target = dr.define_einst(
        t[a, b],
        4 * x[a, c] * u[c, b] + 2 * x[a, c] * v[c, b]
        - 2 * y[a, c] * u[c, b] - y[a, c] * v[c, b]
    )
    targets = [target]

    # The actual optimization.
    res = optimize(targets)
    assert len(res) == 3
