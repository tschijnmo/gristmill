"""Test optimization of trivial tensors."""

from drudge import Drudge
from sympy import symbols

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
