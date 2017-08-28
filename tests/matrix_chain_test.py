"""
Test of the single-term optimization based on matrix chain product.

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

    dr.set_dumms(m_range, symbols('a b c'))
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
