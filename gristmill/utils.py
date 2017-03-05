"""General utilities."""

import collections

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
