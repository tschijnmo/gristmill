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
