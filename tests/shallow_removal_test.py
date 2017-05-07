"""Test of the removal of shallow intermediates."""

from drudge import Range, Drudge
from sympy import symbols, IndexedBase, Symbol, Rational

from gristmill import optimize, verify_eval_seq


def test_removal_of_shallow_interms(spark_ctx):
    """Test removal of shallow intermediates.

    Here we have two intermediates,

    .. math::

        U X V + U Y V

    and

    .. math::

        U X W + U Y W

    and it has been deliberately made such that the multiplication with U should
    be carried out first.  Then after the collection of U, we have a shallow
    intermediate U (X + Y), which is a sum of a single product intermediate.
    This test succeeds when we have two intermediates only without the shallow
    ones.

    """

    # Basic context setting-up.
    dr = Drudge(spark_ctx)

    n = Symbol('n')
    r_large = Range('L', 0, n)
    r_small = Range('S', 0, Rational(1 / 2) * n)

    dumms = symbols('a b c d')
    a, b, c = dumms[:3]
    dumms_small = symbols('e f g h')
    e = dumms_small[0]
    dr.set_dumms(r_large, dumms)
    dr.set_dumms(r_small, dumms_small)
    dr.add_resolver_for_dumms()

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
        s[a, b],
        u[a, c] * x[c, b] + u[a, c] * y[c, b]
    )
    targets = [dr.define_einst(
        t[a, b],
        s_def[a, e] * v[e, b]
    ), dr.define_einst(
        t[a, b],
        s_def[a, e] * w[e, b]
    )]

    # The actual optimization.
    res = optimize(targets)
    assert len(res) == 4

    # Test the correctness.
    assert verify_eval_seq(res, targets, simplify=False)
