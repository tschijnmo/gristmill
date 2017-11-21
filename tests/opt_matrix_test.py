"""
Test of the basic optimization functionality by basic matrix problems.

Matrices are the simplest tensors.  Here we have simple matrix examples that are
very easy to think about.  All the core optimization strategies should first be
tested here.

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
    dr.set_dumms(n_range, symbols('i j k l m n'))
    dr.set_dumms(l_range, symbols('p q r'))
    dr.add_resolver_for_dumms()
    dr.set_name(m, n, l)

    dr.substs = {
        n: m * 2,
        l: m * 3
    }

    return dr


#
# Test of core functionality
# --------------------------
#


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
    stats = {}
    eval_seq = optimize(targets, substs=dr.substs, stats=stats)
    assert stats['Number of nodes'] < 2 ** 3
    assert len(eval_seq) == 2

    # Check the correctness.
    assert verify_eval_seq(eval_seq, targets)

    # Check the cost.
    cost = get_flop_cost(eval_seq)
    leading_cost = get_flop_cost(eval_seq, leading=True)
    expected_cost = 2 * l * m * n + 2 * m ** 2 * n
    assert cost == expected_cost
    assert leading_cost == expected_cost


def test_shallow_matrix_factorization(three_ranges):
    """Test a shallow matrix multiplication factorization problem.

    In this test, there are four matrices involved, X, Y, U, and V.  The final
    expression to optimize is mathematically

    .. math::

        (2 X - Y) * (2 U + V)

    Here, the expression is to be given in its expanded form originally, and
    we test if it can be factorized into something similar to what we have
    above. Here we have the signs and coefficients to have better code
    coverage for these cases.  This test case more concentrates on the
    horizontal complexity in the input.

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


def test_deep_matrix_factorization(three_ranges):
    """Test a basic matrix multiplication factorization problem.

    Similar to the shallow factorization test, the final expression to optimize
    is mathematically

    .. math::

        (X - 2 Y) * U * V

    Different from the shallow test case, here we concentrate more on the
    treatment of depth complexity in the input.  The sum intermediate needs to
    be factored again.

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


def test_factorization_of_two_products(three_ranges):
    """Test a sum where we have two disjoint products.

    The final expression to optimize is

    .. math::

        2 X (3 U + 5 V) - 7 Y (11 U + 13 V) + 17 T

    In this test case, we concentrate on the handling of multiple disjoint
    possible factorization inside a single sum.

    """

    #
    # Basic context setting-up.
    #

    dr = three_ranges
    p = dr.names

    m = p.m
    a, b, c = p.a, p.b, p.c

    # The indexed bases.
    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    # The target.
    target = dr.define_einst(
        IndexedBase('r')[a, b],
        6 * x[a, c] * u[c, b] + 10 * x[a, c] * v[c, b]
        - 77 * y[a, c] * u[c, b] - 91 * y[a, c] * v[c, b]
        + 17 * t[a, b]
    )
    targets = [target]

    # The actual optimization.
    res = optimize(targets)
    assert len(res) == 3
    assert res[-1].n_terms == 3

    # Test the correctness.
    assert verify_eval_seq(res, targets, simplify=True)

    # Test the cost.
    cost = get_flop_cost(res)
    assert cost == 4 * m ** 3 + 4 * m ** 2


def test_general_matrix_problem(three_ranges):
    """Test optimization of a very general matrix computation.

    This is a very general problem trying to test and illustrate many different
    aspects of the optimization, parenthesization, recursion to newly-formed
    factors, and sum of disjoint factorizations.  The target to evaluate reads

    .. math::

        (A + 2B) (3C + 5D) (7E + 13F) + (17P + 19Q) (23X + 29Y)

    where

    - A, B, P, Q is over ranges M, L
    - C, D is over M, N
    - E, F is over N, L
    - X, Y is over L, N

    """

    dr = three_ranges
    p = dr.names

    m, n, l = p.m, p.n, p.l
    a, b = p.a, p.b
    i = p.i
    p = p.p

    f1 = IndexedBase('A')[a, i] + 2 * IndexedBase('B')[a, i]
    f2 = 3 * IndexedBase('C')[i, p] + 5 * IndexedBase('D')[i, p]
    f3 = 7 * IndexedBase('E')[p, b] + 13 * IndexedBase('F')[p, b]
    f4 = 17 * IndexedBase('P')[a, i] + 19 * IndexedBase('Q')[a, i]
    f5 = 23 * IndexedBase('X')[i, b] + 29 * IndexedBase('Y')[i, b]

    target = dr.define_einst(
        IndexedBase('R')[a, b],
        (f1 * f2 * f3 + f4 * f5).expand()
    )
    targets = [target]
    assert target.n_terms == 12
    assert get_flop_cost(targets).subs(dr.substs) == (
        144 * m ** 4 + 16 * m ** 3 + 11 * m ** 2
    )

    eval_seq = optimize(targets, substs=dr.substs)

    # Check the correctness.
    assert verify_eval_seq(eval_seq, targets)
    assert len(eval_seq) == 7
    cost = get_flop_cost(eval_seq)
    assert cost.subs(dr.substs) == 20 * m ** 3 + 16 * m ** 2


#
# Test of special cases
# ---------------------
#


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


def test_optimization_of_common_terms(three_ranges):
    """Test optimization of common terms in summations.

    In this test, there are just two matrices involved, X, Y.  The target reads

    .. math::

        T[a, b] = X[a, b] - X[b, a] + 2 Y[a, b] - 2 Y[b, a]

    Ideally, it should be evaluated as,

    .. math::

        I[a, b] = X[a, b] + 2 Y[a, b]
        T[a, b] = I[a, b] - I[b, a]

    or,

    .. math::

        I[a, b] = X[a, b] - 2 Y[b, a]
        T[a, b] = I[a, b] - I[b, a]

    """

    #
    # Basic context setting-up.
    #
    dr = three_ranges
    p = dr.names

    a, b, c, d = p.a, p.b, p.c, p.d

    # The indexed bases.
    x = IndexedBase('x')
    y = IndexedBase('y')
    t = dr.define_einst(
        IndexedBase('t')[a, b],
        x[a, b] - x[b, a] + 2 * y[a, b] - 2 * y[b, a]
    )

    targets = [t]
    eval_seq = optimize(targets)
    assert len(eval_seq) == 2
    verify_eval_seq(eval_seq, targets)

    # Check the result when the common symmetrization optimization is disabled.
    eval_seq = optimize(targets, opt_symm=False)
    assert len(eval_seq) == 1
    verify_eval_seq(eval_seq, targets)


def test_eval_compression(three_ranges):
    """Test compression of optimized evaluations.

    Here we have two targets,

    .. math::

        U X V + U Y V

    and

    .. math::

        U X W + U Y W

    and it has been deliberately made such that the multiplication with U
    should be carried out first.  Then after the factorization of U, we have
    an intermediate U (X + Y), which is a sum of a single product
    intermediate.  This test succeeds when we have two intermediates only,
    without the unnecessary addition of a single product.

    """

    # Basic context setting-up.
    dr = three_ranges
    p = dr.names

    a = p.a  # Small range
    i, j, k = p.i, p.j, p.k  # Big range

    # The indexed bases.
    u = IndexedBase('U')
    v = IndexedBase('V')
    w = IndexedBase('W')
    x = IndexedBase('X')
    y = IndexedBase('Y')

    s = IndexedBase('S')
    t = IndexedBase('T')

    # The target.
    s_def = dr.define_einst(
        s[i, j],
        u[i, k] * x[k, j] + u[i, k] * y[k, j]
    )
    targets = [dr.define_einst(
        t[i, j],
        s_def[i, a] * v[a, j]
    ), dr.define_einst(
        t[i, j],
        s_def[i, a] * w[a, j]
    )]

    # The actual optimization.
    res = optimize(targets, substs=dr.substs)
    assert len(res) == 4

    # Test the correctness.
    assert verify_eval_seq(res, targets, simplify=False)
