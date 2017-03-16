"""Test the optimization with disconnected outer products."""

from drudge import Range, Drudge
from sympy import symbols, IndexedBase, Symbol

from gristmill import optimize, verify_eval_seq, get_flop_cost


def test_disconnected_outer_product_factorization(spark_ctx):
    """Test optimization of expressions with disconnected outer products.
    """

    # Basic context setting-up.
    dr = Drudge(spark_ctx)

    n = Symbol('n')
    r = Range('R', 0, n)

    dumms = symbols('a b c d e f g')
    a, b, c, d, e = dumms[:5]
    dr.set_dumms(r, dumms)
    dr.add_resolver_for_dumms()

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
    assert cost == 4 * n ** 2
    assert leading_cost == 4 * n ** 2
