"""Test optimization of different special kinds of tensors."""

from drudge import Drudge, Range
from sympy import symbols, IndexedBase, conjugate

from gristmill import optimize, verify_eval_seq


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
    a, b, c = symbols('a b c')
    dr.set_dumms(r, [a, b, c])
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
