"""
Test of the basic functionality of gristmill by basic matrix problems.

Matrices are the simplest tensors.  Here we have simple matrix examples that are
very easy to think about.
"""

import pytest
from drudge import Range, Drudge
from sympy import symbols, IndexedBase

from gristmill import optimize, verify_eval_seq, get_flop_cost


@pytest.fixture
def three_ranges(spark_ctx):
    """Fixture with three ranges.

    This drudge has three ranges, named M, N, L with sizes m, n, and l,
    respectively.  It also has a substitution dictionary setting n = 2m and l
    = 3m.

    """

    dr = Drudge(spark_ctx)

    # The sizes.
    m, n, l = symbols('m n l')

    # The ranges.
    m_range = Range('M', 0, m)
    n_range = Range('N', 0, n)
    l_range = Range('L', 0, l)

    dr.set_dumms(m_range, symbols('a b c d e f g'))
    dr.set_dumms(n_range, symbols('i j k'))
    dr.set_dumms(l_range, symbols('p q r'))
    dr.add_resolver_for_dumms()
    dr.set_name(m, n, l)

    dr.substs = {
        n: m * 2,
        l: m * 3
    }

    return dr


def test_matrix_chain(three_ranges):
    """Test a basic matrix chain multiplication problem.

    Here a very simple matrix chain multiplication problem with three
    matrices are used to test the factorization facilities.  In this simple
    test, we will have three matrices :math:`x`, :math:`y`, and :math:`z`,
    which are of shapes :math:`m\\times n`, :math:`n \\times l`, and :math:`l
    \\times m` respectively. In the factorization, we are going to set
    :math:`n = 2 m` and :math:`l = 3 m`.

    """

    dr = three_ranges
    p = dr.names
    m, n, l = p.m, p.n, p.l

    # The indexed bases.
    x = IndexedBase('x', shape=(m, n))
    y = IndexedBase('y', shape=(n, l))
    z = IndexedBase('z', shape=(l, m))

    target_base = IndexedBase('t')
    target = dr.define_einst(
        target_base[p.a, p.b],
        x[p.a, p.i] * y[p.i, p.p] * z[p.p, p.b]
    )

    # Perform the factorization.
    targets = [target]
    eval_seq = optimize(targets, substs=dr.substs)
    assert len(eval_seq) == 2

    # Check the correctness.
    assert verify_eval_seq(eval_seq, targets)

    # Check the cost.
    cost = get_flop_cost(eval_seq)
    leading_cost = get_flop_cost(eval_seq, leading=True)
    expected_cost = 2 * l * m * n + 2 * m ** 2 * n
    assert cost == expected_cost
    assert leading_cost == expected_cost


def test_matrix_chain_with_sum(three_ranges):
    """Test a matrix chain multiplication with sums.

    This test has a matrix chain multiplication where each of the factors are
    actually a sum of two matrices.

    """

    dr = three_ranges
    p = dr.names
    m, n, l = p.m, p.n, p.l

    # The indexed bases.
    x = dr.define(
        IndexedBase('x')[p.a, p.i],
        IndexedBase('x1')[p.a, p.i] + IndexedBase('x2')[p.a, p.i]
    )
    y = dr.define(
        IndexedBase('y')[p.i, p.p],
        IndexedBase('y1')[p.i, p.p] + IndexedBase('y2')[p.i, p.p]
    )
    z = dr.define(
        IndexedBase('z')[p.p, p.b],
        IndexedBase('z1')[p.p, p.b] + IndexedBase('z2')[p.p, p.b]
    )

    target = dr.define_einst(
        IndexedBase('t')[p.a, p.b],
        x[p.a, p.i] * y[p.i, p.p] * z[p.p, p.b]
    )
    targets = [target]

    eval_seq = optimize(targets, substs=dr.substs)

    # Check the correctness.
    assert verify_eval_seq(eval_seq, targets)
    assert len(eval_seq) == 5
    leading_cost = get_flop_cost(eval_seq, leading=True)
    mult_cost = 2 * l * m * n + 2 * m ** 2 * n
    assert leading_cost == mult_cost
    cost = get_flop_cost(eval_seq)
    assert cost == mult_cost + m * n + n * l + l * m


def test_matrix_factorization(three_ranges):
    """Test a basic matrix multiplication factorization problem.

    In this test, there are four matrices involved, X, Y, U, and V.  And they
    are used in two test cases for different scenarios.

    """

    #
    # Basic context setting-up.
    #

    dr = three_ranges
    p = dr.names

    m = p.m
    a, b, c, d = p.a, p.b, p.c, p.d

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
    assert cost == 2 * m ** 3 + 2 * m ** 2
    assert leading_cost == 2 * m ** 3
    cost = get_flop_cost(res, ignore_consts=False)
    assert cost == 2 * m ** 3 + 4 * m ** 2

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
    assert cost == 4 * m ** 3 + m ** 2
    assert leading_cost == 4 * m ** 3
    cost = get_flop_cost(res, ignore_consts=False)
    assert cost == 4 * m ** 3 + 2 * m ** 2

    # Test disabling summation optimization.
    res = optimize(targets, opt_sum=False)
    assert verify_eval_seq(res, targets, simplify=True)
    new_cost = get_flop_cost(res, ignore_consts=False)
    assert new_cost - cost != 0


def test_disconnected_outer_product_factorization(three_ranges):
    """Test optimization of expressions with disconnected outer products.
    """

    dr = three_ranges
    p = dr.names

    m = p.m
    a, b, c, d, e = p.a, p.b, p.c, p.d, p.e

    # The indexed bases.
    u = IndexedBase('U')
    x = IndexedBase('X')
    y = IndexedBase('Y')
    z = IndexedBase('Z')
    t = IndexedBase('T')

    # The target.
    target = dr.define_einst(
        t[a, b],
        u[a, b] * z[c, e] * x[e, c] + u[a, b] * z[c, e] * y[e, c]
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
    assert cost == 4 * m ** 2
    assert leading_cost == 4 * m ** 2
