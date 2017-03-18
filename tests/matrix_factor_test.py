"""Test of matrix factorization."""

from drudge import Range, Drudge
from sympy import symbols, IndexedBase, Symbol

from gristmill import optimize, verify_eval_seq, get_flop_cost


def test_matrix_factorization(spark_ctx):
    """Test a basic matrix multiplication factorization problem.

    In this test, there are four matrices involved, X, Y, U, and V.  And they
    are used in two test cases for different scenarios.

    """

    #
    # Basic context setting-up.
    #

    dr = Drudge(spark_ctx)

    n = Symbol('n')
    r = Range('R', 0, n)

    dumms = symbols('a b c d e f g')
    a, b, c, d = dumms[:4]
    dr.set_dumms(r, dumms)
    dr.add_resolver_for_dumms()

    # The indexed bases.
    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    #
    # Test case 1.
    #
    # The final expression to optimize is mathematically
    #
    # .. math::
    #
    #     (2 X - Y) * (2 U + V)
    #
    # Here, the expression is to be given in its extended form originally, and
    # we test if it can be factorized into something similar to what we have
    # above. Here we have the signs and coefficients to have better code
    # coverage for these cases.  This test case more concentrates on the
    # horizontal complexity in the input.
    #

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

    # Test the correctness.
    assert verify_eval_seq(res, targets, simplify=False)

    # Test the cost.
    cost = get_flop_cost(res)
    leading_cost = get_flop_cost(res, leading=True)
    assert cost == 2 * n ** 3 + 2 * n ** 2
    assert leading_cost == 2 * n ** 3
    cost = get_flop_cost(res, ignore_consts=False)
    assert cost == 2 * n ** 3 + 4 * n ** 2

    #
    # Test case 2.
    #
    # The final expression to optimize is mathematically
    #
    # .. math::
    #
    #     (X - 2 Y) * U * V
    #
    # Different from the first test case, here we concentrate more on the
    # treatment of depth complexity in the input.  The sum intermediate needs to
    # be factored again.
    #

    # The target.
    target = dr.define_einst(
        t[a, b], x[a, c] * u[c, d] * v[d, b] - 2 * y[a, c] * u[c, d] * v[d, b]
    )
    targets = [target]

    # The actual optimization.
    res = optimize(targets)
    assert len(res) == 3

    # Test the correctness.
    assert verify_eval_seq(res, targets, simplify=True)

    # Test the cost.
    cost = get_flop_cost(res)
    leading_cost = get_flop_cost(res, leading=True)
    assert cost == 4 * n ** 3 + n ** 2
    assert leading_cost == 4 * n ** 3
    cost = get_flop_cost(res, ignore_consts=False)
    assert cost == 4 * n ** 3 + 2 * n ** 2
