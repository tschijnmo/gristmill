"""Test optimization of different special kinds of tensors."""

from drudge import Drudge, Range
from sympy import symbols, IndexedBase, conjugate

from gristmill import optimize, verify_eval_seq, get_flop_cost


def test_simple_scalar_optimization(spark_ctx):
    """Test optimization of a simple scalar.

    There is not much optimization that can be done for simple scalars.  But we
    need to ensure that we get correct result here.
    """

    dr = Drudge(spark_ctx)

    a, b, r = symbols('a b r')
    targets = [dr.define(r, a * b)]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_conjugation_optimization(spark_ctx):
    """Test optimization of expressions containing complex conjugate.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    a, b, c, d = symbols('a b c d')
    dr.set_dumms(r, [a, b, c, d])
    dr.add_default_resolver(r)

    p = IndexedBase('p')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')

    targets = [dr.define_einst(
        p[a, b], x[a, c] * conjugate(y[c, b]) + x[a, c] * conjugate(z[c, b])
    )]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_optimization_handles_coeffcients(spark_ctx):
    """Test optimization of scalar intermediates scaled by coefficients.

    This test comes from PoST theory.  It tests the optimization of tensor
    evaluations with scalar intermediates scaled by a factor.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    a, b = symbols('a b')
    dr.set_dumms(r, [a, b])
    dr.add_default_resolver(r)

    r = IndexedBase('r')
    eps = IndexedBase('epsilon')
    t = IndexedBase('t')

    targets = [dr.define(r[a, b], dr.sum(
        2 * eps[a] * t[a, b]
    ) - 2 * eps[b] * t[a, b])]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_optimization_handles_scalar_intermediates(spark_ctx):
    """Test optimization of scalar intermediates scaling other tensors.

    This is set as a special test primarily since it would entail the same
    collectible giving residues with different ranges.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    dumms = symbols('a b c d e')
    dr.set_dumms(r, dumms)
    a, b, c = dumms[:3]
    dr.add_default_resolver(r)

    u = IndexedBase('u')
    eps = IndexedBase('epsilon')
    t = IndexedBase('t')
    s = IndexedBase('s')

    targets = [dr.define(
        u, (a, r), (b, r),
        dr.sum((c, r), 8 * s[a, b] * eps[c] * t[a])
        - 8 * s[a, b] * eps[a] * t[a]
    )]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_optimization_handles_nonlinear_factors(spark_ctx):
    """Test optimization of with nonlinear factors.

    Here a factor is the square of an indexed quantity.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    dumms = symbols('a b c d e f g h')
    dr.set_dumms(r, dumms)
    a, b, c, d = dumms[:4]
    dr.add_default_resolver(r)

    u = symbols('u')
    s = IndexedBase('s')

    targets = [dr.define(u, dr.sum(
        (a, r), (b, r), (c, r), (d, r),
        32 * s[a, c] ** 2 * s[b, d] ** 2 +
        32 * s[a, c] * s[a, d] * s[b, c] * s[b, d]
    ))]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_common_summation_intermediate_recognition(spark_ctx):
    """Test recognition of summation intermediate differing only in a scalar.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    dumms = symbols('a b c d e f g h')
    dr.set_dumms(r, dumms)
    a, b, c = dumms[:3]
    dr.add_default_resolver(r)

    x = IndexedBase('x')
    y = IndexedBase('y')
    p = IndexedBase('p')
    q = IndexedBase('q')
    r = IndexedBase('r')
    s = IndexedBase('s')

    alpha = symbols('alpha')

    for c1, c2, c3, c4 in [
        (1, 1, 1, 1),
        (1, 1, 2, 2),
        (1, 1, -1, -1),
        (1, -2, -1, 2),
        (1, -1, -1, 1),
        (1, -alpha, 2, -2 * alpha)
    ]:
        targets = [
            dr.define_einst(
                r[a, b],
                c1 * p[a, c] * x[c, b] + c2 * p[a, c] * y[c, b]
            ),
            dr.define_einst(
                s[a, b],
                c3 * q[a, c] * x[c, b] + c4 * q[a, c] * y[c, b]
            )
        ]

        eval_seq = optimize(targets)

        assert verify_eval_seq(eval_seq, targets)
        assert len(eval_seq) == 3


def test_get_cost_on_zero_cost(spark_ctx):
    """Test correct behaviour of get_flop_cost at input with no FLOP cost.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    dumms = symbols('a b')
    dr.set_dumms(r, dumms)
    a, b = dumms[:3]
    dr.add_default_resolver(r)

    x = IndexedBase('x')
    r = IndexedBase('y')

    targets = [
        dr.define_einst(x[a, b], r[a, b])
    ]

    for i in [
        get_flop_cost(targets),
        get_flop_cost(targets, leading=True)
    ]:
        assert i == 0
        continue
