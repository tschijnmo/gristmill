"""Test for optimization of common terms in summations."""

from drudge import Drudge, Range
from sympy import Symbol, symbols, IndexedBase

from gristmill import optimize, get_flop_cost, verify_eval_seq


def test_optimization_of_common_terms(spark_ctx):
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

    Here, in order to emulate real cases where common term reference is in
    interplay with factorization, the X and Y matrices are written as :math:`X =
    S U` and :math:`Y = S V`.

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
    s = IndexedBase('S')
    u = IndexedBase('U')
    v = IndexedBase('V')

    x = dr.define(IndexedBase('X')[a, b], s[a, c] * u[c, b])
    y = dr.define(IndexedBase('Y')[a, b], s[a, c] * v[c, b])
    t = dr.define_einst(
        IndexedBase('t')[a, b],
        x[a, b] - x[b, a] + 2 * y[a, b] - 2 * y[b, a]
    )

    targets = [t]
    eval_seq = optimize(targets)
    assert len(eval_seq) == 3

    # Check correctness.
    verify_eval_seq(eval_seq, targets)

    # Check cost.
    cost = get_flop_cost(eval_seq)
    assert cost == 2 * n ** 3 + 3 * n ** 2
