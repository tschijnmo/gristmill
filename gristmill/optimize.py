"""Optimizer for the contraction computations."""

import collections
import heapq
import itertools
import types
import typing
import warnings

from drudge import TensorDef, prod_, Term, Range, sum_
from sympy import (
    Integer, Symbol, Expr, IndexedBase, Mul, Indexed, sympify, primitive, Wild,
    default_sort_key, oo
)
from sympy.utilities.iterables import multiset_partitions

from .utils import (
    get_cost_key, is_positive_cost, get_total_size, DSF
)


#
#  The public driver
#  -----------------
#


class Strategy:
    """The optimization strategy for tensor contractions.

    This class holds possible options for different aspects of the optimization
    strategy for tensor contractions.  Options for different aspects of the
    problem should be combined by using the bitwise-or ``|`` operator.

    For the optimization of the single-term contractions, we have

    ``GREEDY``
        The contraction will be optimized greedily.  This should only be used
        for large inputs where the other strategies cannot finish within a
        reasonable time.

    ``BEST``
        The global minimum of each tensor contraction will be found by the
        advanced algorithm in gristmill.  And only the optimal contraction(s)
        will be kept for the summation optimization.

    ``SEARCHED``
        The same strategy as ``BEST`` will be attempted for the optimization of
        contractions.  But all evaluations searched in the optimization process
        will be kept and considered in subsequent summation optimizations.

    ``ALL``
        All possible contraction sequences will be considered for all
        contractions.  This can be extremely slow.  But it might be helpful for
        manageable problems.

    For the summation factorization, we have

    ``SUM``
        Factorize the summations in the result.

    ``NOSUM``
        Do not factorize the summations in the result.

    For the common factor optimization, we have

    ``COMMON``
        Skip computation of the same factor up to permutation of indices in
        summations.

    ``NOCOMMON``
        Do not give special treatment for common terms in summation.

    We also have the default optimization strategy as ``DEFAULT``, which will be
    ``SEARCHED | SUM | COMMON``.

    """

    GREEDY = 0
    BEST = 1
    SEARCHED = 2
    ALL = 3

    SUM = 1 << 2
    NOSUM = 0

    COMMON = 1 << 3
    NOCOMMON = 0

    DEFAULT = SEARCHED | SUM | COMMON

    PROD_MASK = 0b11
    MAX = 1 << 4


def optimize(
        computs: typing.Iterable[TensorDef], substs=None, interm_fmt='tau^{}',
        simplify=True, strategy=Strategy.DEFAULT
) -> typing.List[TensorDef]:
    """Optimize the valuation of the given tensor contractions.

    This function will transform the given computations, given as tensor
    definitions, into another list computations mathematically equivalent to the
    given computation while requiring less floating-point operations (FLOPs).

    Parameters
    ----------

    computs
        The computations, can be given as an iterable of tensor definitions.

    substs
        A dictionary for making substitutions inside the sizes of ranges.  All
        the ranges need to have size in at most one undetermined variable after
        the substitution so that they can be totally ordered.

    interm_fmt
        The format for the names of the intermediates.

    simplify
        If the input is going to be simplified before processing.  It can be
        disabled when the input is already simplified.

    strategy
        The optimization strategy, as explained in :py:class:`Strategy`.

    """

    substs = {} if substs is None else substs

    computs = [
        i.simplify() if simplify else i.reset_dumms()
        for i in computs
    ]
    if len(computs) == 0:
        raise ValueError('No computation is given!')

    if not isinstance(strategy, int) or strategy >= Strategy.MAX:
        raise TypeError('Invalid optimization strategy', strategy)

    opt = _Optimizer(
        computs, substs=substs, interm_fmt=interm_fmt, strategy=strategy
    )

    return opt.optimize()


#
# The internal optimization engine
# --------------------------------
#
# General small type definitions and functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# These named tuples should be upgraded when PySpark has support for Python 3.6
# in their stable version.
#
# For general optimization.
#


_Grain = collections.namedtuple('_Grain', [
    'base',
    'exts',
    'terms'
])

_IntermRef = collections.namedtuple('_IntermRef', [
    'coeff',
    'base',
    'indices',
    'power'
])


def _get_ref_from_interm_ref(self: _IntermRef):
    """Get the reference to intermediate without coefficient."""
    return _index(self.base, self.indices) ** self.power


_IntermRef.ref = property(_get_ref_from_interm_ref)

# Symbol/range pairs.
#
# This type is mostly for the convenience of annotation.

_SrPairs = typing.Sequence[typing.Tuple[Symbol, Range]]

#
# Internals for summation and product optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Summation optimization.
#

_LEFT = 0
_RIGHT = 1
_OPPOS = {
    _LEFT: _RIGHT,
    _RIGHT: _LEFT
}

# For type annotation, actually is should be ``_LEFT | _RIGHT`` in Haskell
# algebraic data type notation.

_LR = int

_Ranges = collections.namedtuple('_Ranges', [
    'exts',
    'sums'
])

# Additional information about a node when it is used to augment the current
# biclique.
_NodeInfo = collections.namedtuple('_NodeInfo', [
    'coeff',
    'terms',
    'exc_cost'
])

# Dictionary of the nodes that can possibly to used to augment the current
# biclique.  To be used for variables like ``subg`` and ``cand`` in the
# Bron-Kerbosch algorithm.
_Nodes = typing.Dict[
    typing.Tuple[int, Term], typing.Optional[_NodeInfo]
]

_Edge = collections.namedtuple('_Edge', [
    'term',
    'eval_',
    'coeff',
    'exc_cost'
])

_Biclique = collections.namedtuple('_Biclique', [
    'nodes',  # Left and right.
    'leading_coeff',
    'terms',
    'saving'
])

# These coefficients cached here can make the computation of the saving of a
# biclique easy and fast.

_CostCoeffs = collections.namedtuple('_CostCoeffs', [
    # The final cost for contraction and make an addition of the results.
    'final',
    # The cost of making an addition for left and right factors.
    'preps'
])


def _get_cost_coeffs(ranges: _Ranges) -> _CostCoeffs:
    """Get the cost coefficients for the given ranges."""

    ext_size = get_total_size(itertools.chain.from_iterable(
        ranges.exts
    ))

    final = _get_prod_final_cost(
        ext_size, get_total_size(ranges.sums)
    ) + ext_size

    preps = tuple(
        get_total_size(itertools.chain(i, ranges.sums))
        for i in ranges.exts
    )

    return _CostCoeffs(final=final, preps=preps)


_Saving = collections.namedtuple('_Saving', [
    # Total current saving.
    'saving',
    # Additional saving when one more left/right factor is collected.
    'deltas'
])


def _get_collect_saving(coeffs: _CostCoeffs, n_s: typing.Sequence[int]):
    """Get the saving for collection.

    For the given ranges, when we make a collection of the given number of left
    factors and the given number of right factors, we have saving,

    .. math::

        n_l n_r C(s) e_l e_r s + (n_l n_r - 1) e_l e_r
        - (n_l - 1) e_l s - (n_r - 1) e_r s - C(s) e_l e_r s

    where :math:`C(s)` equals one for no summation and two for the presence of
    summations.  It also equals

    .. math::

        (n_l n_r - 1) (C(s) e_l e_r s + e_l e_r)
        - (n_l - 1) e_l s - (n_r - 1) e_r s

    When we collect terms with :math:`n_l`, it reads,

    .. math::

        n_l (
            n_r C(s) e_l e_r s + n_r e_l e_r - e_l s
        )
        - n_r e_r s
        + e_l s + e_r s - e_l e_r - C(s) e_l e_r s

    or symmetrically

    .. math::

        n_r (
            n_l C(s) e_l e_r s + n_l e_l e_r - e_r s
        )
        - n_l e_l s
        + e_l s + e_r s - e_l e_r - C(s) e_l e_r s

    """

    assert len(n_s) == 2
    assert len(coeffs.preps) == 2

    n_terms = prod_(n_s)

    saving = (n_terms - _UNITY) * coeffs.final - sum(
        (i - _UNITY) * j
        for i, j in zip(n_s, coeffs.preps)
    )

    deltas = []
    # for i, j in zip(reversed(n_s), coeffs.preps):
    for i, v in enumerate(coeffs.preps):
        o = 1 if i == 0 else 0
        if n_s[i] == 0:
            # This could allow bicliques empty in a direction to be augmented by
            # any left or right term.  A value of infinity has to be used to
            # mask the possible non-zero excess costs.
            deltas.append(oo)
        elif n_s[o] == 0:
            # This prevents a dimension get expanded without anything.
            deltas.append(-oo)
        else:
            deltas.append(n_s[o] * coeffs.final - v)
        continue

    return _Saving(saving=saving, deltas=tuple(deltas))


class _BronKerbosch:
    """Iterable for the maximal bicliques."""

    def __init__(self, adjs, ranges):
        """Initialize the iterator."""

        # Static data during the recursion.
        self._adjs = adjs
        self._cost_coeffs = _get_cost_coeffs(ranges)

        # Dynamic data during the recursion.
        #
        # Nodes and coefficients, for left and right.
        self._curr = (
            ([], []),
            ([], [])
        )
        # The set of terms currently in the biclique.
        self._terms = set()
        # The stack of excess costs.
        self._exc_costs = []
        # The leading coefficient.
        self._leading_coeff = None

    def __iter__(self):
        """Iterate over the maximal bicliques."""

        # All left and right nodes.
        nodes = {
            (i, j): _NodeInfo(coeff=_UNITY, terms=set(), exc_cost=0)
            for i, v in enumerate(self._adjs)
            for j in v.keys()
        }

        assert len(nodes) > 0

        yield from self._expand(nodes, dict(nodes), dict(nodes), dict(nodes))

        # If things all goes correctly, the stack should be reverted to initial
        # state by now.
        for i in self._curr:
            for j in i:
                assert len(j) == 0
                continue
            continue

        assert len(self._terms) == 0
        assert len(self._exc_costs) == 0
        assert self._leading_coeff is None

        return

    def _is_expandable(
            self, colour: _LR, node: Term
    ) -> typing.Optional[_NodeInfo]:
        """Test if the given node can currently be expandable.

        When it is expandable, the relevant node information will be
        returned, or None will be the result.

        THIS FUNCTION IS DEPRECATED AND PENDING REMOVAL.  Currently it is only
        used for the sanity checking of the optimized result.
        """

        # Cache frequently used information.
        oppos_colour = _OPPOS[colour]
        adjs = self._adjs[colour][node]
        curr = self._curr
        terms = self._terms

        base_coeff = None
        exc_cost = 0
        new_terms = set()

        for i, v in enumerate(curr[oppos_colour][0]):
            if v not in adjs:
                return
            edge = adjs[v]

            if edge.term in terms or edge.term in new_terms:
                return

            coeff = edge.coeff
            if i == 0:
                base_coeff = coeff
            ratio = coeff / base_coeff
            if ratio != curr[oppos_colour][1][i]:
                return

            exc_cost += edge.exc_cost
            new_terms.add(edge.term)

            continue

        # When we get here, it should be expandable now.
        if self._leading_coeff is None:
            coeff = _UNITY
        else:
            coeff = base_coeff / self._leading_coeff

        # For empty stack, we always get here with base information (coeff=1,
        # new_terms=empty, exc_cost=0).
        return _NodeInfo(coeff=coeff, terms=new_terms, exc_cost=exc_cost)

    def _expand(
            self, subg: _Nodes, curr_subg: _Nodes,
            cand: _Nodes, curr_cand: _Nodes
    ):
        """Generate the bicliques from the current state.

        This is the core of the Bron-Kerbosch algorithm.
        """

        adjs = self._adjs
        exc_costs = self._exc_costs

        # The current state.
        curr = self._curr
        terms = self._terms

        # The code here are adapted from the code in NetworkX for maximal clique
        # problem of simple general graphs.  The original code are kept as much
        # as possible and put in comments.  The original code on which the code
        # is based can be found at,
        #
        # https://github.com/networkx/networkx/blob
        # /48f4b5736174844c77044fae90e3e7adf1dabc10/networkx/algorithms
        # /clique.py#L277-L299
        #

        #
        # u = max(subg, key=lambda u: len(cand & adj[u]))
        #
        # Here only nodes guaranteed to be profitable can be used as the pivot.
        # Or the proof will not work out.

        # Recursion is stopped earlier than here.
        assert len(curr_subg) > 0

        pivot_color, pivot_node = max(curr_subg.keys(), key=lambda x: sum(
            1 for i in adjs[x[0]][x[1]]
            if (_OPPOS[x[0]], i) in cand
        ))

        pivot_oppos = _OPPOS[pivot_color]
        pivot_adj = self._adjs[pivot_color][pivot_node]
        to_loop = (
            (k, v) for k, v in curr_cand.items()
            if k[0] != pivot_oppos or k[1] not in pivot_adj
        )

        #
        # for q in cand - adj[u]:
        #
        to_loop = list(to_loop)
        for q, node_info in to_loop:

            #
            # cand.remove(q)
            #
            colour, node = q
            assert q in cand
            del cand[q]

            #
            # Q.append(q)
            #
            curr[colour][0].append(node)
            curr[colour][1].append(node_info.coeff)
            assert terms.isdisjoint(node_info.terms)
            terms |= node_info.terms
            exc_costs.append(node_info.exc_cost)

            oppos = _OPPOS[colour]
            if len(curr[colour][0]) == 1 and len(curr[oppos][0]) > 0:
                leading_edge = self._adjs[colour][node][
                    curr[oppos][0][0]
                ]
                assert self._leading_coeff is None
                self._leading_coeff = leading_edge.coeff

            ns, saving = self._count_stack()

            #
            # adj_q = adj[q]
            # subg_q = subg & adj_q
            #
            subg_q, curr_subg_q = self._filter_nodes(
                subg, saving, colour, node
            )

            #
            # if not subg_q:
            #    yield Q[:]
            #
            if len(curr_subg_q) == 0:

                # These cases cannot possibly give saving.
                if_skip = any(i == 0 for i in ns) or all(
                    i == 1 for i in ns
                )

                if not if_skip:
                    # The total saving.
                    saving = saving.saving - sum_(exc_costs)

                    if is_positive_cost(saving):
                        yield _Biclique(
                            nodes=tuple(i for i, _ in curr),
                            leading_coeff=self._leading_coeff,
                            terms=terms, saving=saving
                        )

            else:
                #
                # cand_q = cand & adj_q
                #
                cand_q, curr_cand_q = self._filter_nodes(
                    cand, saving, colour, node
                )

                # if cand_q:
                #     for clique in expand(subg_q, cand_q):
                #         yield clique

                if len(curr_cand_q) > 0:
                    yield from self._expand(
                        subg_q, curr_subg_q, cand_q, curr_cand_q
                    )

            #
            # Q.pop()
            #
            for i in curr[colour]:
                i.pop()
            terms -= node_info.terms
            exc_costs.pop()
            if len(curr[colour][0]) == 0:
                self._leading_coeff = None

    def _filter_nodes(
            self, nodes: _Nodes,
            saving: _Saving, new_colour: _LR, new_node: Term
    ) -> typing.Tuple[_Nodes, _Nodes]:
        """Filter the nodes for the current stack.

        In the original Bron-Kerbosch algorithm, both subg and cand are filtered
        by union with the adjacent nodes of the newly added node.  Now the
        computation can be a lot more complex than that.  We need to note,

        1. No term already contained can be decomposed in another way in a
        different evaluation.

        2. The coefficients need to match the existing proportion.

        We also have less to note in that we do not require any connectivity
        among nodes with the same colour.

        Here all expandable nodes and the profitable ones among them for the
        current step will be returned.  The profitable nodes for the current
        step contains only the nodes that is profitable right now.  The all
        expandable nodes has all nodes that are valid to be augmented into the
        current stack.
        """

        curr = self._curr

        all_ = {}
        for k, v in nodes.items():
            # The node with the same colour as the new node will not be affected
            # by the new addition.
            colour, node = k
            if colour == new_colour:
                all_[k] = v
                continue

            adj = self._adjs[colour][node]
            if new_node not in adj:
                continue
            edge = adj[new_node]

            assert _OPPOS[colour] == new_colour
            oppos_curr = curr[new_colour]
            # We have at least the new node was just added.
            assert len(oppos_curr[0]) > 0
            assert len(oppos_curr[1]) == len(oppos_curr[0])

            leading_edge = adj[oppos_curr[0][0]]
            ratio = edge.coeff / leading_edge.coeff

            if ratio != oppos_curr[1][-1]:
                continue

            new_coeff = (
                v.coeff if self._leading_coeff is None
                else leading_edge.coeff / self._leading_coeff
            )

            new_terms = set(v.terms)
            new_terms.add(edge.term)
            if not new_terms.isdisjoint(self._terms):
                continue

            new_exc_cost = v.exc_cost + edge.exc_cost

            node_info = _NodeInfo(
                coeff=new_coeff, terms=new_terms, exc_cost=new_exc_cost
            )

            # Sanity checking, should be disabled in production.
            assert node_info == self._is_expandable(colour, node)

            all_[k] = node_info
            continue

        curr = {k: v for k, v in all_.items() if is_positive_cost(
            saving.deltas[k[0]] - v.exc_cost
        )}

        return all_, curr

    def _count_stack(self):
        """Count the current size of the stack.

        The saving will also be returned.
        """
        ns = tuple(
            len(self._curr[i][0]) for i in [_LEFT, _RIGHT]
        )
        saving = _get_collect_saving(self._cost_coeffs, ns)
        return ns, saving


class _CollectGraph:
    """Graph for the collectibles of a given range.

    This data structure, and the maximal biclique generation in Bron-Kerbosch
    style, are the core of the factorization algorithm for sums.

    We have separate graph for different ranges.  For each range, the graph has
    the factors as nodes, and actual evaluations with the factors as edges.
    Internally, the graph is stored as two sparse adjacent lists.
    """

    def __init__(self):
        """Initialize the collectible table."""
        self._adjs = (
            collections.defaultdict(dict),
            collections.defaultdict(dict)
        )

    def add(self, left, right, term, eval_, coeff, exc_cost):
        """Add a new edge to the graph."""

        edge = _Edge(
            term=term, eval_=eval_, coeff=coeff, exc_cost=exc_cost
        )

        left_adj = self._adjs[_LEFT][left]
        assert right not in left_adj
        left_adj[right] = edge

        right_adj = self._adjs[_RIGHT][right]
        assert left not in right_adj
        right_adj[left] = edge

    def gen_bicliques(
            self, ranges: _Ranges
    ) -> typing.Iterable[_Biclique]:
        """Generate the bicliques within the graph.

        For performance reasons, the bicliques generated will contain references
        to internal mutable data.  It is the responsibility of the caller to
        make proper copy when it is necessary.
        """

        yield from _BronKerbosch(self._adjs, ranges)

    def remove_terms(self, terms: typing.AbstractSet[int]) -> bool:
        """Remove all edges and nodes involving the given terms.

        If a value of True is returned, we have an empty graph after the
        removal.
        """

        new_adjs = (
            collections.defaultdict(dict),
            collections.defaultdict(dict)
        )
        if_empty = True

        for old, new in zip(self._adjs, new_adjs):
            for from_node, conns in old.items():
                new_conns = {
                    to_node: edge
                    for to_node, edge in conns.items()
                    if edge.term not in terms
                }
                if len(new_conns) > 0:
                    if_empty = False
                    new[from_node] = new_conns
                continue
            continue

        self._adjs = new_adjs

        return if_empty


_Collectibles = typing.DefaultDict[_Ranges, _CollectGraph]

#
# For product optimization.
#

_Part = collections.namedtuple('_Part', [
    'ref',
    'node'
])


def _get_prod_final_cost(exts_total_size, sums_total_size) -> Expr:
    """Compute the final cost for a pairwise product evaluation."""

    if sums_total_size == 1:
        return exts_total_size
    else:
        return _TWO * exts_total_size * sums_total_size


#
# Core evaluation DAG nodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#


class _EvalNode:
    """A node in the evaluation graph.
    """

    def __init__(self, base: Symbol, exts: _SrPairs):
        """Initialize the evaluation node.
        """

        self.base = base
        self.exts = exts

        # For optimization.
        self.evals = []  # type: typing.List[_EvalNode]
        self.total_cost = None

        # For result finalization.
        self.n_refs = 0
        self.generated = False

    def get_substs(self, indices):
        """Get substitutions and symbols requiring exclusion before indexing.

        First resetting dummies excluding the returned symbols and then making
        the returned substitution on each term could achieve indexing.  Since
        the real free symbols are already gather from all inputs, the free
        symbols are not considered here.  But they should be added for the
        resetting.
        """

        substs = {}
        excl = set()

        assert len(indices) == len(self.exts)
        for i, j in zip(indices, self.exts):
            dumm = j[0]
            substs[dumm] = i
            excl.add(dumm)
            excl |= i.atoms(Symbol)
            continue

        return substs, excl


class _Sum(_EvalNode):
    """Sum nodes in the evaluation graph."""

    def __init__(self, base, exts, sum_terms):
        """Initialize the node."""
        super().__init__(base, exts)
        self.sum_terms = sum_terms

    def __repr__(self):
        """Form a representation string for the node."""
        return '_Sum(base={}, exts={}, sum_terms={})'.format(
            repr(self.base), repr(self.exts), repr(self.sum_terms)
        )


class _Prod(_EvalNode):
    """Product nodes in the evaluation graph.
    """

    def __init__(self, base, exts, sums, coeff, factors):
        """Initialize the node."""
        super().__init__(base, exts)
        self.sums = sums
        self.coeff = coeff
        self.factors = factors

    def __repr__(self):
        """Form a representation string for the node."""
        return '_Prod(base={}, exts={}, sums={}, coeff={}, factors={})'.format(
            repr(self.base), repr(self.exts), repr(self.sums),
            repr(self.coeff), repr(self.factors)
        )


#
# Core optimizer class
# ~~~~~~~~~~~~~~~~~~~~
#


class _Optimizer:
    """Optimizer for tensor contraction computations.

    This internal optimizer can only be used once for one set of input.
    """

    #
    # Public functions.
    #

    def __init__(self, computs, substs, interm_fmt, strategy):
        """Initialize the optimizer."""

        # Information to be read from the input computations.
        #
        # The only drudge for the inputs.
        self._drudge = None
        # The only variable for range sizes.
        self._range_var = None
        # Mapping from the substituted range to original range.
        self._input_ranges = {}
        # Symbols that should not be used for dummies.
        self._excl = set()

        # Read, process, and verify user input.
        self._grist = [
            self._form_grain(comput, substs) for comput in computs
        ]

        # Dummies stock in terms of the substituted range.
        assert self._drudge is not None
        self._dumms = {
            k: self._drudge.dumms.value[v]
            for k, v in self._input_ranges.items()
        }

        # Other internal data preparation.
        self._interm_fmt = interm_fmt
        self._strategy = strategy

        self._next_internal_idx = 0

        # From intermediate base to actual evaluation node.
        self._interms = {}
        # From the canonical form to intermediate base.
        self._interms_canon = {}

        self._res = None

    def optimize(self):
        """Optimize the evaluation of the given computations.
        """

        if self._res is not None:
            return self._res

        res_nodes = [self._form_node(i) for i in self._grist]
        for i in res_nodes:
            self._optimize(i)
            continue

        self._res = self._linearize(res_nodes)
        return self._res

    #
    # User input pre-processing.
    #

    def _form_grain(self, comput, substs):
        """Form grain for a given computation.
        """

        curr_drudge = comput.rhs.drudge
        if self._drudge is None:
            self._drudge = curr_drudge
        elif self._drudge is not curr_drudge:
            raise ValueError(
                'Invalid computations to optimize, containing two drudges',
                (self._drudge, curr_drudge)
            )
        else:
            pass

        # Externals processing.
        exts = self._proc_sums(comput.exts, substs)
        ext_symbs = {i for i, _ in exts}

        # Terms processing.
        terms = []
        for term in comput.rhs_terms:
            if not term.is_scalar:
                raise ValueError(
                    'Invalid term to optimize', term, 'expecting scalar'
                )
            sums = self._proc_sums(term.sums, substs)
            amp = term.amp

            # Add the true free symbols to the exclusion set.
            self._excl |= term.free_vars - ext_symbs
            terms.append(Term(sums, amp, ()))

            continue

        return _Grain(
            base=comput.base if len(exts) == 0 else comput.base.args[0],
            exts=exts, terms=terms
        )

    def _proc_sums(self, sums, substs):
        """Process a summation list.

        The ranges will be replaced with substituted sizes.  Relevant members of
        the optimizer will also be updated.  User error will also be reported.
        """

        res = []
        for symb, range_ in sums:

            if not range_.bounded:
                raise ValueError(
                    'Invalid range for optimization', range_,
                    'expecting explicit bound'
                )
            lower, upper = [
                self._check_range_var(i.xreplace(substs), range_)
                for i in [range_.lower, range_.upper]
            ]

            new_range = Range(range_.label, lower=lower, upper=upper)
            if new_range not in self._input_ranges:
                self._input_ranges[new_range] = range_
            elif range_ != self._input_ranges[new_range]:
                raise ValueError(
                    'Invalid ranges', (range_, self._input_ranges[new_range]),
                    'duplicated labels'
                )
            else:
                pass

            res.append((symb, new_range))
            continue

        return tuple(res)

    def _check_range_var(self, expr, range_) -> Expr:
        """Check size expression for valid symbol presence."""

        range_vars = expr.atoms(Symbol)
        if len(range_vars) == 0:
            pass
        elif len(range_vars) == 1:

            range_var = range_vars.pop()
            if self._range_var is None:
                self._range_var = range_var
            elif self._range_var != range_var:
                raise ValueError(
                    'Invalid range', range_, 'unexpected symbol',
                    range_var, 'conflicting with', self._range_var
                )
            else:
                pass
        else:
            raise ValueError(
                'Invalid range', range_, 'containing multiple symbols',
                range_vars
            )

        return expr

    #
    # Optimization result post-processing.
    #

    def _linearize(
            self, optimized: typing.Sequence[_EvalNode]
    ) -> typing.List[TensorDef]:
        """Linearize optimized forms of the evaluation.
        """

        for node in optimized:
            self._set_n_refs(node)
            continue

        # Separate the intermediates and the results so that the results can be
        # guaranteed to be at the end of the evaluation sequence.
        interms = []
        res = []
        for node in optimized:
            curr = self._linearize_node(node, interms, keep=True)
            assert curr is not None
            res.append(curr)
            continue

        return self._finalize(itertools.chain(interms, res))

    def _set_n_refs(self, node: _EvalNode):
        """Set reference counts from an evaluation node.

        It is always the first evaluation that is going to be used, all rest
        will be removed to avoid further complications.
        """

        if len(node.evals) == 0:
            self._optimize(node)
        assert len(node.evals) > 0
        del node.evals[1:]
        eval_ = node.evals[0]

        if isinstance(eval_, _Prod):
            possible_refs = [i for i in eval_.factors]
        elif isinstance(eval_, _Sum):
            possible_refs = eval_.sum_terms
        else:
            assert False

        for i in possible_refs:
            ref = self._parse_interm_ref(i)
            if ref is None:
                continue
            dep_node = self._interms[ref.base]
            dep_node.n_refs += 1
            self._set_n_refs(dep_node)
            continue

        return

    def _linearize_node(self, node: _EvalNode, res: list, keep=False):
        """Linearize evaluation rooted in the given node into the result.

        If keep if set to True, the evaluation of the given node will not be
        appended to the result list.
        """

        if node.generated:
            return None

        def_, deps = self._form_def(node)
        for i in deps:
            self._linearize_node(self._interms[i], res)
            continue

        node.generated = True

        if not keep:
            res.append(def_)

        return def_

    def _form_def(self, node: _EvalNode):
        """Form the final definition of an evaluation node.

        The dependencies will also be returned.
        """

        assert len(node.evals) == 1

        if isinstance(node, _Prod):
            return self._form_prod_def(node)
        elif isinstance(node, _Sum):
            return self._form_sum_def(node)
        else:
            assert False

    def _form_prod_def(self, node: _Prod):
        """Form the final definition of a product evaluation node."""

        exts = node.exts
        eval_ = node.evals[0]
        assert isinstance(eval_, _Prod)
        term, deps = self._form_prod_def_term(eval_)
        return _Grain(
            base=node.base, exts=exts, terms=[term]
        ), deps

    def _form_prod_def_term(self, eval_: _Prod):
        """Form the term in the final definition of a product evaluation node.
        """

        amp = eval_.coeff

        deps = []
        for factor in eval_.factors:

            ref = self._parse_interm_ref(factor)
            if ref is not None:
                assert ref.coeff == 1
                interm = self._interms[ref.base]
                if self._is_input(interm):
                    # Inline trivial reference to an input.
                    content = self._get_content(factor)
                    assert len(content) == 1
                    assert len(content[0].sums) == 0
                    amp *= content[0].amp ** ref.power
                else:
                    deps.append(ref.base)
                    amp *= factor
            else:
                amp *= factor
        return Term(eval_.sums, amp, ()), deps

    def _form_sum_def(self, node: _Sum):
        """Form the final definition of a sum evaluation node."""

        exts = node.exts
        terms = []
        deps = []

        eval_ = node.evals[0]
        assert isinstance(eval_, _Sum)

        sum_terms = []
        self._inline_sum_terms(eval_.sum_terms, sum_terms)
        for term in sum_terms:

            ref = self._parse_interm_ref(term)
            if ref is None:
                terms.append(Term((), term, ()))
                # No dependency for pure scalars.
                continue

            assert ref.power == 1  # Higher power not possible in sum.

            # Sum term are guaranteed to be formed from references to products,
            # never directly written in terms of input.
            term_node = self._interms[ref.base]

            if term_node.n_refs == 1 or self._is_input(term_node):
                # Inline intermediates only used here and simple input
                # references.

                eval_ = term_node.evals[0]
                assert isinstance(eval_, _Prod)
                contents = self._index_prod(eval_, ref.indices)
                assert len(contents) == 1
                term = contents[0]
                factors, term_coeff = term.get_amp_factors(self._interms)

                # Switch back to evaluation node for using the facilities for
                # product nodes.
                tmp_node = _Prod(
                    term_node.base, exts, term.sums,
                    ref.coeff * term_coeff, factors
                )
                new_term, term_deps = self._form_prod_def_term(tmp_node)

                terms.append(new_term)
                deps.extend(term_deps)

            else:
                terms.append(Term(
                    (), term, ()
                ))
                deps.append(ref.base)
            continue

        return _Grain(
            base=node.base, exts=exts, terms=terms
        ), deps

    def _inline_sum_terms(
            self, sum_terms: typing.Sequence[Expr], res: typing.List[Expr]
    ):
        """Inline the summation terms from single-reference terms.

        This function mutates the given result list rather than returning the
        result to avoid repeated list creation in recursive calls.
        """

        for sum_term in sum_terms:
            ref = self._parse_interm_ref(sum_term)
            if ref is None:
                res.append(sum_term)
                continue
            assert ref.power == 1

            node = self._interms[ref.base]
            assert len(node.evals) > 0
            eval_ = node.evals[0]

            if_inline = isinstance(eval_, _Sum) and (
                node.n_refs == 1 or len(eval_.sum_terms) == 1
            )
            if if_inline:
                if len(node.exts) == 0:
                    substs = None
                else:
                    substs = {
                        i[0]: j for i, j in zip(eval_.exts, ref.indices)
                    }

                proced_sum_terms = [
                    (
                        i.xreplace(substs) if substs is not None else sum_term
                    ) * ref.coeff for i in eval_.sum_terms
                ]
                self._inline_sum_terms(proced_sum_terms, res)
                continue
            else:
                res.append(sum_term)
            continue

        return

    def _is_input(self, node: _EvalNode):
        """Test if a product node is just a trivial reference to an input."""
        if isinstance(node, _Prod):
            return len(node.sums) == 0 and len(node.factors) == 1 and (
                self._parse_interm_ref(node.factors[0]) is None
            )
        else:
            return False

    def _finalize(
            self, computs: typing.Iterable[_Grain]
    ) -> typing.List[TensorDef]:
        """Finalize the linearization result.

        Things will be cast to drudge tensor definitions, with intermediates
        holding names formed from the format given by user.
        """

        next_idx = 0

        substs = {}  # For normal substitution of bases.
        repls = []  # For removed shallow intermediates

        def proc_amp(amp):
            """Process the amplitude by making the found substitutions."""
            for i in reversed(repls):
                amp = amp.replace(*i)
                continue
            return amp.xreplace(substs)

        res = []
        for comput in computs:
            exts = tuple((s, self._input_ranges[r]) for s, r in comput.exts)
            if_scalar = len(exts) == 0
            base = comput.base if if_scalar else IndexedBase(comput.base)

            terms = [
                i.map(proc_amp, sums=tuple(
                    (s, self._input_ranges[r]) for s, r in i.sums
                )) for i in comput.terms
            ]

            # No internal intermediates should be leaked.
            for i in terms:
                assert not any(j in self._interms for j in i.free_vars)

            if comput.base in self._interms:

                if len(terms) == 1 and len(terms[0].sums) == 0:
                    # Remove shallow intermediates.  The saving might be too
                    # modest to justify the additional memory consumption.
                    #
                    # TODO: Move it earlier to a better place.
                    repl_lhs = base if if_scalar else base[tuple(
                        _WILD_FACTORY[i] for i, _ in enumerate(exts)
                    )]
                    repl_rhs = proc_amp(terms[0].amp.xreplace(
                        {v[0]: _WILD_FACTORY[i] for i, v in enumerate(exts)}
                    ))
                    repls.append((repl_lhs, repl_rhs))
                    continue  # No new intermediate added.

                final_base = (
                    Symbol if if_scalar else IndexedBase
                )(self._interm_fmt.format(next_idx))
                next_idx += 1
                substs[base] = final_base
            else:
                final_base = base

            res.append(TensorDef(
                final_base, exts, self._drudge.create_tensor(terms)
            ).reset_dumms())
            continue

        return res

    #
    # Internal support utilities.
    #

    def _get_next_internal(self):
        """Get the symbol for the next internal intermediate.
        """
        idx = self._next_internal_idx
        self._next_internal_idx += 1
        return Symbol('gristmillInternalIntermediate{}'.format(idx))

    @staticmethod
    def _write_in_orig_ranges(sums):
        """Write the summations in terms of undecorated bare ranges.

        The labels in the ranges are assumed to be decorated.
        """
        return tuple(
            (i, j.replace_label(j.label[0])) for i, j in sums
        )

    def _canon_terms(self, new_sums: _SrPairs, terms: typing.Iterable[Term]):
        """Form a canonical label for a list of terms.

        The new summation list is prepended to the summation list of all terms.
        The coefficient ahead of the canonical form is returned before the
        canonical form.  And the permuted new summation list is also returned
        after the canonical form.  Note that this list contains the original
        dummies given in the new summation list, while the terms has reset new
        dummies.

        Note that the ranges in the new summation list are assumed to be
        decorated with labels earlier than _SUMMED.  In the result, they are
        still in decorated forms and are guaranteed to be permuted in the same
        way for all given terms.  The summations from the terms will be
        internally decorated but written in bare ranges in the final result.

        Note that this is definitely a poor man's version of canonicalization of
        multi-term tensor definitions with external indices.  A lot of cases
        cannot be handled well.  Hopefully it can be replaced with a systematic
        treatment some day in the future.

        """

        new_dumms = {i for i, _ in new_sums}
        coeffs = []

        candidates = collections.defaultdict(list)
        for term in terms:
            term, canon_sums = self._canon_term(new_sums, term)

            factors, coeff = term.amp_factors
            coeffs.append(coeff)

            candidates[
                term.map(lambda x: prod_(factors))
            ].append(canon_sums)
            continue

        # Poor man's canonicalization of external indices.
        #
        # This algorithm is not guaranteed to work.  Here we just choose an
        # ordering of the external indices that is as safe as possible.  But
        # certainly it is not guaranteed to work for all cases.
        #
        # TODO: Fix it!

        chosen = min(candidates.items(), key=lambda x: (
            len(x[1]), -len(x[0].amp.atoms(Symbol) & new_dumms),
            x[0].sort_key
        ))

        canon_new_sums = set(chosen[1])
        if len(canon_new_sums) > 1:
            warnings.warn(
                'Internal deficiency: '
                'summation intermediate may not be fully canonicalized'
            )
        # This could also fail when the chosen term has symmetry among the new
        # summations not present in any other term.  This can be hard to check.

        canon_new_sum = canon_new_sums.pop()
        preferred = chosen[0].amp_factors[1]
        canon_coeff = _get_canon_coeff(coeffs, preferred)

        res_terms = []
        for term in terms:
            canon_term, _ = self._canon_term(canon_new_sum, term, fix_new=True)
            # TODO: Add support for complex conjugation.
            res_terms.append(canon_term.map(lambda x: x / canon_coeff))
            continue

        return canon_coeff, tuple(
            sorted(res_terms, key=lambda x: x.sort_key)
        ), canon_new_sum

    def _canon_term(self, new_sums, term, fix_new=False):
        """Canonicalize a single term.

        Internal method for _canon_terms, not supposed to be directly called.
        """

        term = Term(tuple(itertools.chain(
            (
                (v[0], v[1].replace_label((v[1].label[0], _EXT, i)))
                for i, v in enumerate(new_sums)
            ) if fix_new else new_sums,
            (
                (i, j.replace_label((j.label, _SUMMED)))
                for i, j in term.sums
            )
        )), term.amp, ())
        canoned = term.canon(symms=self._drudge.symms.value)

        canon_sums = canoned.sums
        canon_orig_sums = self._write_in_orig_ranges(canon_sums)

        dumm_reset, _ = canoned.map(sums=canon_orig_sums).reset_dumms(
            dumms=self._dumms, excl=self._excl
        )

        canon_new_sums = []
        term_new_sums = []
        term_sums = []
        i_new = 0
        for i, j in zip(dumm_reset.sums, canon_sums):
            if j[1].label[1] == _SUMMED:
                # Existing summations.
                term_sums.append(i)
            else:
                if fix_new:
                    assert j[0] == new_sums[i_new][0]
                    range_ = new_sums[i_new][1]
                else:
                    range_ = j[1]
                canon_new_sums.append((j[0], range_))
                term_new_sums.append((i[0], range_))
                i_new += 1
            continue

        return dumm_reset.map(sums=tuple(itertools.chain(
            term_new_sums, term_sums
        ))), tuple(canon_new_sums)

    def _parse_interm_ref(self, expr: Expr) -> typing.Optional[_IntermRef]:
        """Parse an expression that is possibly an intermediate reference.
        """

        coeff = _UNITY
        base = None
        indices = None
        power = None

        if isinstance(expr, Mul):
            args = expr.args
        else:
            args = [expr]

        for i in args:
            if any(j in self._interms for j in i.atoms(Symbol)):
                assert base is None
                ref, power = i.as_base_exp()
                if isinstance(ref, Indexed):
                    base = ref.base.args[0]
                    indices = ref.indices
                elif isinstance(ref, Symbol):
                    base = ref
                    indices = ()
                else:
                    assert False
                assert base in self._interms
            else:
                coeff *= i

        return None if base is None else _IntermRef(
            coeff=coeff, base=base, indices=indices, power=power
        )

    def _get_content(self, interm_ref: Expr) -> typing.List[Term]:
        """Get the content of an intermediate reference.

        This function might be removed after the new factorization algorithm is
        implemented.
        """

        ref = self._parse_interm_ref(interm_ref)
        assert ref is not None

        node = self._interms[ref.base]

        if isinstance(node, _Sum):
            content = self._index_sum(node, ref.indices)
        elif isinstance(node, _Prod):
            content = self._index_prod(node, ref.indices)
        else:
            assert False

        return [
            i.scale(ref.coeff) for i in self._raise_power(content, ref.power)
        ]

    def _index_sum(self, node: _Sum, indices) -> typing.List[Term]:
        """Substitute the external indices in the sum node"""

        substs, _ = node.get_substs(indices)

        res = []
        for i in node.sum_terms:
            term = i.xreplace(substs)
            ref = self._parse_interm_ref(term)
            if ref is None:
                res.append(term)
            else:
                term_def = self._get_content(term)
                res.extend(term_def)
            continue

        return res

    def _index_prod(self, node: _Prod, indices) -> typing.List[Term]:
        """Substitute the external indices in the evaluation node."""

        substs, excl = node.get_substs(indices)

        term = Term(
            node.sums, node.coeff * prod_(node.factors), ()
        ).reset_dumms(
            self._dumms, excl=self._excl | excl
        )[0].map(lambda x: x.xreplace(substs))

        return [term]

    def _raise_power(
            self, terms: typing.Sequence[Term], exp: int
    ) -> typing.List[Term]:
        """Raise the sum of the given terms to the given power."""
        curr = []  # type: typing.List[Term]
        for _ in range(exp):
            if len(curr) == 0:
                curr = list(terms)
            else:
                # TODO: Make the multiplication more efficient.
                curr = [i.mul_term(
                    j, dumms=self._dumms,
                    excl=self._excl | i.free_vars | j.free_vars
                ) for i, j in itertools.product(curr, terms)]
        return curr

    #
    # General optimization.
    #

    def _form_node(self, grain: _Grain):
        """Form an evaluation node from a tensor definition.

        This is the entry point for optimization.
        """

        # We assume it is fully simplified and expanded by grist preparation.
        exts = grain.exts
        terms = grain.terms

        assert len(terms) > 0
        return self._form_sum_from_terms(
            grain.base, exts, terms
        )

    def _optimize(self, node):
        """Optimize the evaluation of the given node.

        The evaluation methods will be filled with, possibly multiple, method of
        evaluations.
        """

        if len(node.evals) > 0:
            return node

        if isinstance(node, _Sum):
            return self._optimize_sum(node)
        elif isinstance(node, _Prod):
            return self._optimize_prod(node)
        else:
            assert False

    def _form_prod_interm(
            self, exts, sums, factors
    ) -> typing.Tuple[Expr, _EvalNode]:
        """Form a product intermediate.

        The factors are assumed to be all non-trivial factors needing
        processing.
        """

        decored_exts = tuple(
            (i, j.replace_label((j.label, _EXT)))
            for i, j in exts
        )
        n_exts = len(decored_exts)
        term = Term(tuple(sums), prod_(factors), ())

        coeff, key, canon_exts = self._canon_terms(
            decored_exts, [term]
        )
        assert len(key) == 1

        if key in self._interms_canon:
            base = self._interms_canon[key]
        else:
            base = self._get_next_internal()
            self._interms_canon[key] = base

            key_term = key[0]
            key_exts = self._write_in_orig_ranges(key_term.sums[:n_exts])
            key_sums = key_term.sums[n_exts:]
            key_factors, key_coeff = key_term.get_amp_factors(self._interms)
            interm = _Prod(
                base, key_exts, key_sums, key_coeff, key_factors
            )
            self._interms[base] = interm

        return coeff * _index(
            base, canon_exts, strip=True
        ), self._interms[base]

    def _form_sum_interm(
            self, exts: _SrPairs, terms: typing.Sequence[Term]
    ) -> typing.Tuple[Expr, _EvalNode]:
        """Form a sum intermediate.
        """

        decored_exts = tuple(
            (i, j.replace_label((j.label, _EXT)))
            for i, j in exts
        )
        n_exts = len(decored_exts)

        coeff, canon_terms, canon_exts = self._canon_terms(decored_exts, terms)

        if canon_terms in self._interms_canon:
            base = self._interms_canon[canon_terms]
        else:
            base = self._get_next_internal()
            self._interms_canon[canon_terms] = base

            node_exts = None
            node_terms = []
            for term in canon_terms:
                term_exts = self._write_in_orig_ranges(term.sums[:n_exts])
                if node_exts is None:
                    node_exts = term_exts
                else:
                    assert node_exts == term_exts
                node_terms.append(term.map(sums=term.sums[n_exts:]))
                continue

            node = self._form_sum_from_terms(base, node_exts, node_terms)
            self._interms[base] = node
            self._optimize(node)

        return coeff * _index(
            base, canon_exts, strip=True
        ), self._interms[base]

    def _form_sum_from_terms(
            self, base: Symbol, exts: _SrPairs, terms: typing.Iterable[Term]
    ):
        """Form a summation node for given the terms.

        No processing is done in this method.  It just forms the node.
        """
        sum_terms = []
        plain_scalars = []
        ext_symbs = {i for i, _ in exts}
        for term in terms:
            sums = term.sums
            factors, coeff = term.get_amp_factors(ext_symbs)
            if len(factors) == 0:
                plain_scalars.append(coeff)
            else:
                interm_ref, _ = self._form_prod_interm(exts, sums, factors)
                sum_terms.append(interm_ref * coeff)
            continue

        if len(plain_scalars) > 0:
            sum_terms.append(sum_(plain_scalars))

        return _Sum(base, exts, sum_terms)

    #
    # Sum optimization.
    #

    def _optimize_sum(self, sum_node: _Sum):
        """Optimize the summation node."""

        # We first optimize the common terms.
        exts = sum_node.exts
        terms, new_term_idxes = self._optimize_common_terms(sum_node)

        if self._strategy & Strategy.SUM > 0:
            terms = self._factorize_sum(terms, new_term_idxes, exts)

        sum_node.evals = [_Sum(
            sum_node.base, sum_node.exts, terms
        )]
        return

    def _optimize_common_terms(self, sum_node: _Sum) -> typing.Tuple[
        typing.List[Expr], typing.List[int]
    ]:
        """Perform optimization of common intermediate references.
        """

        exts_dict = dict(sum_node.exts)

        # Intermediate base -> (indices -> coefficient)
        #
        # This also gather terms with the same reference to deeper nodes.
        interm_refs = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )

        plain_scalars = []
        for term in sum_node.sum_terms:
            ref = self._parse_interm_ref(term)
            if ref is None:
                plain_scalars.append(term)
                continue
            assert ref.power == 1

            interm_refs[ref.base][ref.indices] += ref.coeff
            continue

        # Intermediate referenced only once goes to the result directly and wait
        # to be factored, others wait to be pulled and do not participate in
        # factorization.
        res_terms = plain_scalars

        if self._strategy & Strategy.COMMON > 0:
            res_collectible_idxes = self._optimize_common_symmtrization(
                interm_refs, exts_dict, res_terms
            )
        else:
            res_collectible_idxes = self._form_interm_refs(
                interm_refs, res_terms
            )

        return res_terms, res_collectible_idxes

    def _optimize_common_symmtrization(self, interm_refs, exts_dict, res_terms):
        """Optimize common symmetrization in the intermediate references.
        """

        res_collectible_idxes = []
        # Indices, coeffs tuple -> base, coeff
        pull_info = collections.defaultdict(list)
        for k, v in interm_refs.items():

            if len(v) == 0:
                assert False
            elif len(v) == 1:
                res_collectible_idxes.append(len(res_terms))
                indices, coeff = v.popitem()
                res_terms.append(
                    _index(k, indices) * coeff
                )
            else:
                # Here we use name for sorting directly, since here we cannot
                # have general expressions hence no need to use the expensive
                # sort_key.
                raw = list(v.items())  # Indices/coefficient pairs.
                raw.sort(key=lambda x: [i.name for i in x[0]])
                leading_coeff = raw[0][1]
                pull_info[tuple(
                    (i, j / leading_coeff) for i, j in raw
                )].append((k, leading_coeff))

        # Now we treat the terms from which new intermediates might be pulled
        # out.
        for k, v in pull_info.items():
            pivot = k[0][0]
            assert len(pivot) > 0
            assert k[0][1] == 1
            if len(v) == 1:
                # No need to form a new intermediate.
                base, coeff = v[0]
                pivot_ref = _index(base, pivot) * coeff
            else:
                # We need to form an intermediate here.
                interm_exts = tuple(
                    (i, exts_dict[i]) for i in pivot
                )
                interm_terms = [
                    term.scale(coeff)
                    for base, coeff in v
                    for term in self._get_content(_index(base, pivot))
                ]
                pivot_ref, interm_node = self._form_sum_interm(
                    interm_exts, interm_terms
                )

                self._optimize(interm_node)

            for indices, coeff in k:
                substs = {
                    i: j for i, j in zip(pivot, indices)
                }
                res_terms.append(
                    pivot_ref.xreplace(substs) * coeff
                )
                continue

            continue
        return res_collectible_idxes

    @staticmethod
    def _form_interm_refs(interm_refs, res_terms):
        """Form intermediate references directly.
        """

        res_collectible_idxes = []

        for k, v in interm_refs.items():

            assert len(v) > 0

            # Only intermediates referenced once could participate in
            # factorization.
            if len(v) == 1:
                res_collectible_idxes.append(len(res_terms))

            for indices, coeff in v.items():
                res_terms.append(
                    _index(k, indices) * coeff
                )

            continue

        return res_collectible_idxes

    def _factorize_sum(self, terms, new_term_idxes, exts):
        """Factorize the summations greedily.
        """

        if_keep = [True for _ in terms]
        new_terms = []

        collectibles = self._find_collectibles(terms, new_term_idxes, exts)
        while True:

            ranges, biclique = self._choose_collectible(collectibles)
            if ranges is None:
                break

            new_terms.append(self._form_factored_term(ranges, biclique))
            self._clean_up_collected(biclique, collectibles, if_keep)

            continue
        # End Main loop.

        new_terms.extend(i for i, j in zip(terms, if_keep) if j)
        return new_terms

    def _find_collectibles(
            self, terms, new_term_idxes, exts
    ) -> _Collectibles:
        """Find all collectibles for the given terms..
        """

        res = collections.defaultdict(_CollectGraph)  # type: _Collectibles

        for term_idx in new_term_idxes:
            term = terms[term_idx]
            ref = self._parse_interm_ref(term)
            if ref is None:
                continue

            node = self._interms[ref.base]
            assert isinstance(node, _Prod)

            self._optimize(node)
            for eval_idx, eval_ in enumerate(node.evals):
                assert isinstance(eval_, _Prod)
                self._find_collectibles_eval(
                    term_idx, ref, eval_idx, eval_, exts, res
                )
                continue

        return res

    def _find_collectibles_eval(
            self, term_idx: int, ref: _IntermRef, eval_idx: int,
            eval_: _Prod, exts: _SrPairs, res: _Collectibles
    ):
        """Get the collectibles for a particular evaluations of a product.
        """

        if len(eval_.factors) < 2:
            return
        assert len(eval_.factors) == 2

        total_cost = eval_.total_cost
        opt_cost = self._interms[ref.base].total_cost
        exc_cost = total_cost - opt_cost

        sums = tuple(sorted(
            eval_.sums, key=lambda x: default_sort_key(x[0])
        ))

        # Get the factoris and coefficients, no need to make substitution when
        # there is no external indices.
        if len(eval_.exts) == 0:
            assert len(ref.indices) == 0
            coeff = ref.coeff * eval_.coeff
            factors = eval_.factors
            assert factors[0] != factors[1]
        else:
            eval_terms = self._index_prod(eval_, ref.indices)
            assert len(eval_terms) == 1
            eval_term = eval_terms[0]
            factors, coeff = eval_term.get_amp_factors(self._interms)
            coeff *= ref.coeff

        excl = self._excl | {i for i, _ in exts}

        # Information about the (two) factors,
        #
        # expr: The original expression for the factor.
        # exts: Indices of the involved externals.
        # canon_content: The canonicalized content for the factor.
        factor_infos = [
            types.SimpleNamespace(expr=i) for i in factors
        ]

        for f_i in factor_infos:
            content = self._get_content(f_i.expr)
            assert len(content) == 1
            content = content[0]

            symbs = f_i.expr.atoms(Symbol)
            f_i.exts = tuple(
                i for i, v in enumerate(exts) if v[0] in symbs
            )  # Index only.

            # In order to really make sure, the content will be re-canonicalized
            # based on the current ambient.
            canon_content = content.canon().reset_dumms(
                self._dumms, excl=excl
            )[0]

            _, canon_coeff = canon_content.get_amp_factors(self._interms)
            f_i.canon_content = canon_content.map(
                lambda x: x / canon_coeff, skip_vecs=True
            )
            coeff *= canon_coeff

            continue

        factor_infos.sort(key=lambda x: x.exts)

        l_exts, r_exts = [
            tuple(exts[j] for j in i.exts)
            for i in factor_infos
        ]
        ranges = _Ranges(exts=(l_exts, r_exts), sums=sums)

        # When the left and right externals differ, the two factors have
        # determined colour, or we need to add one of them for each colour
        # assignment.
        lr_factor_idxes = [(0, 1)]
        if l_exts == r_exts:
            lr_factor_idxes.append((1, 0))
        lr_factors = [
            tuple(factor_infos[j].canon_content for j in i)
            for i in lr_factor_idxes
        ]
        for i in lr_factors:
            res[ranges].add(
                left=i[0], right=i[1],
                term=term_idx, eval_=eval_idx, coeff=coeff, exc_cost=exc_cost
            )
            continue

        return

    @staticmethod
    def _choose_collectible(collectibles: _Collectibles):
        """Choose the most profitable collectible factor.
        """

        best_saving = None
        best_ranges = None
        best_biclique = None
        for ranges, graph in collectibles.items():
            for biclique in graph.gen_bicliques(ranges):

                saving = get_cost_key(biclique.saving)

                if best_saving is None or saving > best_saving:
                    best_saving = saving
                    best_ranges = ranges
                    # Make copy only when we need them.
                    best_biclique = _Biclique(
                        nodes=tuple(tuple(i) for i in biclique.nodes),
                        leading_coeff=biclique.leading_coeff,
                        terms=frozenset(biclique.terms),
                        saving=biclique.saving
                    )

                continue

        return best_ranges, best_biclique

    def _form_factored_term(
            self, ranges: _Ranges, biclique: _Biclique
    ) -> Expr:
        """Form the factored term for the given factorization."""

        leading_coeff = biclique.leading_coeff

        # Form and optimize the two new summation nodes.
        factors = [leading_coeff]
        for exts_i, nodes_i in zip(ranges.exts, biclique.nodes):
            expr, eval_node = self._form_sum_interm(exts_i, [
                i.scale(1 / leading_coeff) for i in nodes_i
            ])
            factors.append(expr)
            self._optimize(eval_node)
            continue

        # Form the contraction node for the two new summation nodes.
        exts = tuple(sorted(
            set(itertools.chain.from_iterable(ranges.exts)),
            key=lambda x: default_sort_key(x[0])
        ))
        expr, eval_node = self._form_prod_interm(
            exts, ranges.sums, factors
        )

        # Make phony optimization of the intermediate.
        eval_node.total_cost = _UNITY
        eval_node.evals = [eval_node]

        return expr

    @staticmethod
    def _clean_up_collected(
            biclique: _Biclique, collectibles: _Collectibles,
            if_keep: typing.List[bool]
    ):
        """Clean up the collectibles and the terms after factorization."""

        to_remove = []
        for ranges, graph in collectibles.items():
            if_empty = graph.remove_terms(biclique.terms)
            if if_empty:
                to_remove.append(ranges)
            continue
        for i in to_remove:
            del collectibles[i]
            continue

        for i in biclique.terms:
            assert if_keep[i]
            if_keep[i] = False

    #
    # Product optimization.
    #

    def _optimize_prod(self, prod_node):
        """Optimize the product evaluation node.
        """

        assert len(prod_node.evals) == 0
        n_factors = len(prod_node.factors)

        if n_factors < 2:
            assert n_factors == 1
            prod_node.evals.append(prod_node)
            prod_node.total_cost = _get_prod_final_cost(
                get_total_size(prod_node.exts),
                get_total_size(prod_node.sums)
            )
            return

        strategy = self._strategy & Strategy.PROD_MASK
        evals = prod_node.evals
        optimal_cost = None
        for final_cost, broken_sums, parts_gen in self._gen_factor_parts(
                prod_node
        ):
            def need_break():
                """If we need to break the current loop."""
                if strategy == Strategy.GREEDY:
                    return True
                elif strategy == Strategy.BEST or strategy == Strategy.SEARCHED:
                    return get_cost_key(final_cost) > optimal_cost[0]
                elif strategy == Strategy.ALL:
                    return False
                else:
                    assert False

            if (optimal_cost is not None) and need_break():
                break
            # Else

            for parts in parts_gen:

                # Recurse, two parts.
                assert len(parts) == 2
                for i in parts:
                    self._optimize(i.node)
                    continue

                total_cost = (
                    final_cost
                    + parts[0].node.total_cost
                    + parts[1].node.total_cost
                )
                total_cost_key = get_cost_key(total_cost)

                if_new_optimal = (
                    optimal_cost is None or optimal_cost[0] > total_cost_key
                )
                if if_new_optimal:
                    optimal_cost = (total_cost_key, total_cost)
                    if strategy == Strategy.BEST:
                        evals.clear()

                # New optimal is always added.
                def need_add_eval():
                    """If the current evaluation need to be added."""
                    if strategy == Strategy.BEST:
                        return total_cost_key == optimal_cost[0]
                    else:
                        return True

                if if_new_optimal or need_add_eval():
                    new_eval = self._form_prod_eval(
                        prod_node, broken_sums, parts
                    )
                    new_eval.total_cost = total_cost
                    evals.append(new_eval)

                continue

        assert len(evals) > 0
        prod_node.total_cost = optimal_cost[1]
        return

    def _gen_factor_parts(self, prod_node: _Prod):
        """Generate all the partitions of factors in a product node."""

        # Compute things invariant to different summations for performance.
        exts = prod_node.exts
        exts_total_size = get_total_size(exts)

        factor_atoms = [
            i.atoms(Symbol) for i in prod_node.factors
        ]
        sum_involve = [
            {j for j, v in enumerate(factor_atoms) if i in v}
            for i, _ in prod_node.sums
        ]

        dumm2index = tuple(
            {v[0]: j for j, v in enumerate(i)}
            for i in [prod_node.exts, prod_node.sums]
        )
        # Indices of external and internal dummies involved by each factors.
        factor_infos = [
            tuple(
                set(i[j] for j in atoms if j in i)
                for i in dumm2index
            )
            for atoms in factor_atoms
        ]

        # Actual generation.
        for broken_size, kept in self._gen_kept_sums(prod_node.sums):
            broken_sums = [i for i, j in zip(prod_node.sums, kept) if not j]
            final_cost = _get_prod_final_cost(
                exts_total_size, broken_size
            )
            yield final_cost, broken_sums, self._gen_parts_w_kept_sums(
                prod_node, kept, sum_involve, factor_infos
            )
            continue

    @staticmethod
    def _gen_kept_sums(sums):
        """Generate kept summations in increasing size of broken summations.

        The results will be given as boolean array giving if the corresponding
        entry is to be kept.
        """

        sizes = [i.size for _, i in sums]
        n_sums = len(sizes)

        def get_size(kept):
            """Wrap the kept summation with its size."""
            size = sympify(prod_(
                i for i, j in zip(sizes, kept) if not j
            ))
            return get_cost_key(size), size, kept

        init = [True] * n_sums  # Everything is kept.
        queue = [get_size(init)]
        while len(queue) > 0:
            curr = heapq.heappop(queue)
            yield curr[1], curr[2]
            curr_kept = curr[2]
            for i in range(n_sums):
                if curr_kept[i]:
                    new_kept = list(curr_kept)
                    new_kept[i] = False
                    heapq.heappush(queue, get_size(new_kept))
                    continue
                else:
                    break
            continue

    def _gen_parts_w_kept_sums(
            self, prod_node: _Prod, kept, sum_involve, factor_infos
    ):
        """Generate all partitions with given summations kept.

        First we the factors are divided into chunks indivisible according to
        the kept summations.  Then their bipartitions which really break the
        broken sums are generated.
        """

        dsf = DSF(i for i, _ in enumerate(factor_infos))

        for i, j in zip(kept, sum_involve):
            if i:
                dsf.union(j)
            continue

        chunks = dsf.sets
        if len(chunks) < 2:
            return

        for part in self._gen_parts_from_chunks(kept, chunks, sum_involve):
            assert len(part) == 2
            yield tuple(
                self._form_part(prod_node, i, sum_involve, factor_infos)
                for i in part
            )

        return

    @staticmethod
    def _gen_parts_from_chunks(kept, chunks, sum_involve):
        """Generate factor partitions from chunks.

        Here special care is taken to respect the broken summations in the
        result.
        """

        n_chunks = len(chunks)

        for chunks_part in multiset_partitions(n_chunks, m=2):
            factors_part = tuple(set(
                factor_i for chunk_i in chunk_part_i
                for factor_i in chunks[chunk_i]
            ) for chunk_part_i in chunks_part)

            for i, v in enumerate(kept):
                if v:
                    continue
                # Now we have broken sum, it need to be involved by both parts.
                involve = sum_involve[i]
                if any(part.isdisjoint(involve) for part in factors_part):
                    break
            else:
                yield factors_part

    def _form_part(self, prod_node, factor_idxes, sum_involve, factor_infos):
        """Form a partition for the given factors."""

        involved_exts, involved_sums = [
            set.union(*[factor_infos[i][label] for i in factor_idxes])
            for label in [0, 1]
        ]

        factors = [prod_node.factors[i] for i in factor_idxes]
        exts = [
            v
            for i, v in enumerate(prod_node.exts)
            if i in involved_exts
        ]
        sums = []

        for i, v in enumerate(prod_node.sums):
            if sum_involve[i].isdisjoint(factor_idxes):
                continue
            elif sum_involve[i] <= factor_idxes:
                sums.append(v)
            else:
                exts.append(v)
            continue

        ref, node = self._form_prod_interm(exts, sums, factors)
        return _Part(ref=ref, node=node)

    def _form_prod_eval(
            self, prod_node: _Prod, broken_sums, parts: typing.Tuple[_Part, ...]
    ):
        """Form an evaluation for a product node."""

        assert len(parts) == 2

        coeff = _UNITY
        factors = []
        for i in parts:
            curr_ref = self._parse_interm_ref(i.ref)
            coeff *= curr_ref.coeff
            factors.append(curr_ref.ref)
            continue

        return _Prod(
            prod_node.base, prod_node.exts, broken_sums,
            coeff * prod_node.coeff, factors
        )


#
# Utility constants.
#


_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)
_TWO = Integer(2)

_EXT = 0
_SUMMED_EXT = 1
_SUMMED = 2

_SUBSTED_EVAL_BASE = Symbol('gristmillSubstitutedEvalBase')


#
# Utility static functions.
#

class _SymbFactory(dict):
    """A small symbol factory."""

    def __missing__(self, key):
        return Symbol('gristmillInternalSymbol{}'.format(key))


_SYMB_FACTORY = _SymbFactory()


class _WildFactory(dict):
    """A small wild symbol factory."""

    def __missing__(self, key):
        return Wild('gristmillInternalWild{}'.format(key))


_WILD_FACTORY = _WildFactory()


def _get_canon_coeff(coeffs, preferred):
    """Get the canonical coefficient from a list of coefficients."""

    coeff, _ = primitive(sum(
        v * _SYMB_FACTORY[i] for i, v in enumerate(coeffs)
    ))

    # The primitive computation does not take phase into account.
    n_neg = 0
    n_pos = 0
    for i in coeffs:
        if i.has(_NEG_UNITY) or i.is_negative:
            n_neg += 1
        else:
            n_pos += 1
        continue
    if n_neg > n_pos:
        phase = _NEG_UNITY
    elif n_pos > n_neg:
        phase = _UNITY
    else:
        preferred_phase = (
            _NEG_UNITY if preferred.has(_NEG_UNITY) or preferred.is_negative
            else _UNITY
        )
        phase = preferred_phase

    return coeff * phase


def _index(base, indices, strip=False) -> Expr:
    """Index the given base with indices.

    When strip is set to true, the indices are assumed to be symbol/range pairs
    list.
    """

    if strip:
        indices = tuple(i for i, _ in indices)
    else:
        indices = tuple(indices)

    return base if len(indices) == 0 else IndexedBase(base)[indices]


#
# Optimization result verification
# --------------------------------
#


def verify_eval_seq(
        eval_seq: typing.Sequence[TensorDef], res: typing.Sequence[TensorDef],
        simplify=False
) -> bool:
    """Verify the correctness of an evaluation sequence for the results.

    The last entries of the evaluation sequence should be in one-to-one
    correspondence with the original form in the ``res`` argument.  This
    function returns ``True`` when the evaluation sequence is symbolically
    equivalent to the given raw form.  When a difference is found,
    ``ValueError`` will be raised with relevant information.

    Note that this function can be very slow for large evaluations.  But it is
    advised to be used for all optimizations in mission-critical tasks.

    Parameters
    ----------

    eval_seq
        The evaluation sequence to verify, can be the output from
        :py:func:`optimize` directly.

    res
        The original result to test the evaluation sequence against.  It can be
        the input to :py:func:`optimize` directly.

    simplify
        If simplification is going to be performed after each step of the
        back-substitution.  It is advised for larger complex evaluations.

    """

    n_res = len(res)
    n_interms = len(eval_seq) - n_res

    substed_eval_seq = []
    defs_dict = {}
    for idx, eval_ in enumerate(eval_seq):
        base = eval_.base
        free_vars = eval_.rhs.free_vars
        curr_defs = [
            defs_dict[i] for i in free_vars if i in defs_dict
        ]
        exts = {i for i, _ in eval_.exts}
        rhs = eval_.rhs.subst_all(curr_defs, simplify=simplify, excl=exts)
        new_def = TensorDef(base, eval_.exts, rhs)
        substed_eval_seq.append(new_def)

        if idx < n_interms:
            defs_dict[
                base.label if isinstance(base, IndexedBase) else base
            ] = new_def

        continue

    for i, j in zip(substed_eval_seq[-n_res:], res):
        ref = j.simplify()
        if i.lhs != ref.lhs:
            raise ValueError(
                'Unequal left-hand sides', i.lhs, 'with', ref.lhs
            )
        diff = (i.rhs - ref.rhs).simplify()
        if diff != 0:
            raise ValueError(
                'Unequal definition for ', j.lhs, j
            )
        continue

    return True
