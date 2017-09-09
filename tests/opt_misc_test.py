"""Test optimization of different special kinds of tensor computations.
"""

import pytest
from drudge import Drudge, Range
from sympy import symbols, IndexedBase, conjugate

from gristmill import optimize, verify_eval_seq, get_flop_cost


@pytest.fixture(scope='module')
def simple_drudge(spark_ctx):
    """Make simple drudge.

    This fixture gives a simple drudge with a simple range and a few dummies.
    """

    dr = Drudge(spark_ctx)

    n = symbols('n')
    r = Range('r', 0, n)
    dumms = symbols('a b c d e f g h')
    dr.set_dumms(r, dumms)
    dr.add_default_resolver(r)

    dr.n = n
    dr.r = r
    dr.ds = dumms

    return dr


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


def test_conjugation_optimization(simple_drudge):
    """Test optimization of expressions containing complex conjugate.
    """

    dr = simple_drudge

    a, b, c, d = dr.ds[:4]

    p = IndexedBase('p')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')

    targets = [dr.define_einst(
        p[a, b], x[a, c] * conjugate(y[c, b]) + x[a, c] * conjugate(z[c, b])
    )]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_optimization_handles_coeffcients(simple_drudge):
    """Test optimization of scalar intermediates scaled by coefficients.

    This test comes from PoST theory.  It tests the optimization of tensor
    evaluations with scalar intermediates scaled by a factor.
    """

    dr = simple_drudge

    a, b = dr.ds[:2]

    r = IndexedBase('r')
    eps = IndexedBase('epsilon')
    t = IndexedBase('t')

    targets = [dr.define(r[a, b], dr.sum(
        2 * eps[a] * t[a, b]
    ) - 2 * eps[b] * t[a, b])]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_optimization_handles_scalar_intermediates(simple_drudge):
    """Test optimization of scalar intermediates scaling other tensors.

    This is set as a special test primarily since it would entail the same
    collectible giving residues with different ranges.
    """

    dr = simple_drudge

    r = dr.r
    a, b, c = dr.ds[:3]

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


def test_optimization_handles_nonlinear_factors(simple_drudge):
    """Test optimization of with nonlinear factors.

    Here a factor is the square of an indexed quantity.
    """

    dr = simple_drudge

    r = dr.r
    a, b, c, d = dr.ds[:4]

    u = symbols('u')
    s = IndexedBase('s')

    targets = [dr.define(u, dr.sum(
        (a, r), (b, r), (c, r), (d, r),
        32 * s[a, c] ** 2 * s[b, d] ** 2 +
        32 * s[a, c] * s[a, d] * s[b, c] * s[b, d]
    ))]
    eval_seq = optimize(targets)
    assert verify_eval_seq(eval_seq, targets)


def test_common_summation_intermediate_recognition(simple_drudge):
    """Test recognition of summation intermediate differing only in a scalar.
    """

    dr = simple_drudge

    a, b, c = dr.ds[:3]

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


def test_removal_of_shallow_interms(simple_drudge):
    """Test if removal of shallow intermediates can be turned on/off."""

    dr = simple_drudge

    r = dr.r
    a, b, c, d = dr.ds[:4]

    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    u = IndexedBase('u')

    targets = [
        dr.define(
            u, (a, r), (b, r), (c, r),
            dr.sum((d, r), x[a, d] * y[b, d] * z[c, d])
        )
    ]

    for i in [True, False]:
        eval_seq = optimize(targets, remove_shallow=i)
        verify_eval_seq(eval_seq, targets)
        assert len(eval_seq) == (
            1 if i else 2
        )
        continue


def test_get_cost_on_zero_cost(simple_drudge):
    """Test correct behaviour of get_flop_cost at input with no FLOP cost.
    """

    dr = simple_drudge

    a, b = dr.ds[:2]

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
