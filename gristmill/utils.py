"""General utilities."""

import collections
import typing

from drudge import prod_, TensorDef
from sympy import Expr, Symbol, Poly, Integer, Mul, poly_from_expr


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


def get_total_size(sums) -> Expr:
    """Get the total size of a summation list."""
    size = prod_(i.size for _, i in sums)
    if isinstance(size, Expr):
        return size
    elif isinstance(size, int):
        return Integer(size)
    else:
        raise TypeError('Invalid total size', size, 'from sums', sums)


#
# Public cost computation function.
#


def get_flop_cost(eval_seq: typing.Iterable[TensorDef], leading=False):
    """Get the FLOP cost for the given evaluation sequence.

    This function gives the count of floating-point operations, addition and
    multiplication, involved by the evaluation sequence.  Note that the cost of
    copying and initialization are not counted.  And this function is only
    applicable where the amplitude of the terms are simple products.

    Parameters
    ----------

    eval_seq
        The evaluation sequence whose FLOP cost is to be estimated.  It should
        be given as an iterable of tensor definitions.

    leading
        If only the cost terms with leading scaling be given.  When multiple
        symbols are present in the range sizes, terms with the highest total
        scaling is going to be picked.

    """

    cost = sum(_get_flop_cost(i) for i in eval_seq)
    return _get_leading(cost) if leading else cost


def _get_flop_cost(step):
    """Get the FLOP cost of a tensor evaluation step."""

    ext_size = get_total_size(step.exts)

    cost = Integer(0)
    n_terms = 0
    for term in step.rhs_terms:
        sum_size = get_total_size(term.sums)

        if isinstance(term.amp, Mul):
            n_mult = len(term.amp.args) - 1
        else:
            n_mult = 0

        if sum_size == 1:
            n_add = 0
        else:
            n_add = 1

        cost = add_costs(cost, (n_add + n_mult) * ext_size * sum_size)
        n_terms += 1
        continue

    if n_terms > 1:
        cost = add_costs(cost, (n_terms - 1) * ext_size)

    return cost


def _get_leading(cost):
    """Get the leading terms in a cost polynomial."""

    symbs = tuple(cost.atoms(Symbol))
    poly, _ = poly_from_expr(cost, *symbs)
    terms = poly.terms()

    leading_deg = max(sum(i) for i, _ in terms)
    leading_cost = sum(
        coeff * prod_(i ** j for i, j in zip(symbs, degs))
        for degs, coeff in terms
        if sum(degs) == leading_deg
    )

    return leading_cost


#
# Disjoint set forest.
#

class DSF(object):
    """
    Disjoint sets forest.

    This is a very simple implementation of the disjoint set forest data
    structure for finding the inseparable chunks of factors for given ranges to
    keep.  Heuristics of union by rank and path compression are both applied.

    Attributes
    ----------

    _contents
        The original contents of the nodes.  The object that was used for
        building the node.  Note that this class is designed with hashable and
        simple things like integers in mind.

    _parents
        The parent of the nodes.  Given as index in the contents list.

    _ranks
        The rank of the nodes.

    _locs
        The dictionary mapping the contents of the nodes into its location in
        this data structure.

    """

    def __init__(self, contents):
        """Initialize the object.

        Parameters
        ----------

        contents
            An iterable of the contents of the nodes.
        """

        self._contents = []
        self._ranks = []
        self._parents = []
        self._locs = {}

        for i, v in enumerate(contents):
            self._contents.append(v)
            self._ranks.append(0)
            self._parents.append(i)
            self._locs[v] = i
            continue

    def union(self, contents):
        """Put the given contents into union.

        Note that the nodes to be unioned are given in terms of their contents
        rather than their index. Also contents missing in the forest will just
        be **ignored**, which is what is needed for the case of inseparable
        chunk finding.

        Parameters
        ----------

        contents
            An iterable of the contents that are going to be put into the same
            set.

        """

        represent = None  # The set that other sets will be unioned to.
        for i in contents:

            try:
                loc = self._locs[i]
            except KeyError:
                continue

            if represent is None:
                represent = loc
            else:
                self._union_idxes(represent, loc)

            continue

        return None

    @property
    def sets(self):
        """The disjoints sets as actual sets.

        This property will convert the disjoint sets in the internal data
        structure into an actual list of sets.  A list of sets will be given,
        where each set contains the contents of the nodes in the subset.
        """

        # A dictionary with the set representative *index* as key and the
        # sets of the *contents* in the subset as values.
        sets_dict = collections.defaultdict(set)

        for i, v in enumerate(self._contents):
            sets_dict[self._find_set(i)].add(v)
            continue

        return list(sets_dict.values())

    def _union_idxes(self, idx1, idx2):
        """Union the subsets that the two given indices are in."""

        set1 = self._find_set(idx1)
        set2 = self._find_set(idx2)
        rank1 = self._ranks[set1]
        rank2 = self._ranks[set2]

        # Union by rank.
        if rank1 > rank2:
            self._parents[set2] = set1
        else:
            self._parents[set1] = set2
            if rank1 == rank2:
                self._ranks[set2] += 1

    def _find_set(self, idx):
        """Find the representative index of the subset the given index is in."""

        parent = self._parents[idx]
        if idx != parent:
            self._parents[idx] = self._find_set(parent)
        return self._parents[idx]
