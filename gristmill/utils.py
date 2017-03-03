"""General utilities."""

from sympy import Expr, Symbol, Poly


#
# Cost-related utilities.
#

def get_cost_key(cost: Expr):
    """Get the key for ordering the cost.

    The cost should be a polynomial of at most one undetermined variable.  The
    result gives ordering of the cost agreeing with our common sense.
    """

    symbs = cost.atoms(Symbol)
    n_symbs = len(symbs)

    if n_symbs == 0:
        return 0, [cost]
    elif n_symbs == 1:
        symb = symbs.pop()
        coeffs = Poly(cost, symb).all_coeffs()
        return len(coeffs) - 1, coeffs
    else:
        raise ValueError(
            'Invalid cost to compare', cost,
            'expecting univariate polynomial or number'
        )


def add_costs(*args):
    """Add the arguments as costs.

    Here when one of the operand is unity, it will be taken as a zero in the
    summation.
    """

    res = sum(i if abs(i) != 1 else 0 for i in args)
    return res if res != 0 else 1
