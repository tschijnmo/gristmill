"""Optimizer for the contraction computations."""

import collections
import enum
import functools
import heapq
import itertools
import operator
import typing
import warnings

from drudge import TensorDef, prod_, Term, Range, sum_
from networkx import Graph
from sympy import (
    Integer, Symbol, Expr, IndexedBase, Mul, Indexed, primitive, Wild,
    default_sort_key, Pow
)

from .utils import (
    Size, get_total_size, DSF, Tuple4Cmp, form_sized_range
)


#
#  The public driver
#  -----------------
#


class ContrStrat(enum.Enum):
    """The strategies for handling tensor contractions.

    This class holds possible options for different ways of handling
    contractions in the optimization, for both the termination of the main
    loop and the retention of parenthesizations for sum optimization.
    Specifically, we have options

    ``GREEDY``
        The contraction within each term will be optimized greedily.  This
        accelerates the optimization with big sacrifice of the result
        quality.  So it should only be used for inputs having terms
        containing many factors by a very dense pattern.

    ``OPT``
        The global minimum of each tensor contraction will be found by the
        advanced algorithm in gristmill.  And only the optimal contraction(s)
        will be kept for the sum optimization.

    ``TRAV``
        The same strategy as ``OPT`` will be attempted for the optimization
        of contractions.  But all evaluations traversed in the optimization
        process will be kept and considered in subsequent summation
        optimizations.

    ``EXHAUST``
        All possible parenthesizations will be considered for all terms. This
        can be extremely slow.  But it might be helpful for problems having
        terms all with manageable number of factors.

    """

    GREEDY = 0
    OPT = 1
    TRAV = 2
    EXHAUST = 3


class RepeatedTermsStrat(enum.Enum):
    """Optimization for repeated terms in a sum.

    In some sum of tensor contractions, some terms might be different components
    of the same computed tensor.  For instance, in

    .. math::

        r_{a, b} = s_a t_b + s_b t_a

    if we define

    .. math::

        i_{a, b} = s_a t_b

    the two terms are actually :math:`i_{a, b}` and :math:`i_{b, a}`.  For
    problem with repeated terms, we have strategies,

    ``SKIP``

        Repeated terms are simply skipped during the optimization by
        factorization.  In this way, repeated terms are guaranteed not to be
        computed twice even implicitly.

    ``NATURAL``

        Repeated terms participates factorization only when faster evaluation is
        given by this.  Technically, this is achieved by setting the excess cost
        of the evaluation of the terms to be the **full cost** of the
        evaluation, rather than the difference with the optimal cost.  This
        setting should give acceptable result for most purposes.

    ``IGNORE``

        Ignore the fact that the terms are repeated.  They are going to be
        treated exactly like other terms.

    """

    SKIP = 0
    NATURAL = 1
    IGNORE = 2


def optimize(computs: typing.Iterable[TensorDef], substs=None, simplify=True,
             interm_fmt='tau^{}', contr_strat=ContrStrat.TRAV, opt_sum=True,
             repeated_terms_strat=RepeatedTermsStrat.NATURAL,
             opt_symm=True, req_an_opt=False, greedy_cutoff=-1, drop_cutoff=-1,
             remove_shallow=True) -> typing.List[TensorDef]:
    """Optimize the evaluation of the given tensor computations.

    This function will transform the given computations, given as tensor
    definitions, into another list of computations mathematically equivalent
    to the given computations, while requiring less arithmetic operations.

    Parameters
    ----------

    computs
        The computations, can be given as an iterable of tensor definitions.

    substs
        A dictionary for making substitutions inside the sizes of ranges.  All
        the ranges need to have size in at most one undetermined variable after
        the substitution, so that they can be totally ordered.  When one symbol
        still remains in the sizes, the asymptotic cost (scaling and prefactor)
        will be optimized.  Or when all symbols are gone after the substitution,
        optimization is going to be based on the numeric sizes.  Numeric sizes
        tend to make the optimization faster due to the usage of built-in
        integer or floating point arithmetic in lieu of the more complex
        polynomial arithmetic.

    simplify
        If the input is going to be simplified before processing.  It can be
        disabled when the input is already simplified.

    interm_fmt
        The format for the names of the intermediates.

    contr_strat
        The strategy for handling contractions, as explained in
        :py:class:`ContrStrat`.

    repeated_terms_strat
        The strategy for handling repeated terms in sums, as explained in
        :py:class:`RepeatedTermsStrat`.

    opt_sum
        If sums of multiple terms will be attempted to be optimized by using
        constriction (factorization).

    opt_symm
        If common symmetrization of multiple tensors, input or intermediate,
        is going to be optimized.  For instance, with it, :math:`x_{a,
        b} + y_{a, b} - 2 x_{b, a} - 2 y_{b, a}` can be optimized into first
        computing :math:`p_{a, b} = x_{a, b} + y_{a, b}` followed by
        :math:`p_{a, b} - 2 p_{b, a}`.

    req_an_opt
        If each constriction operation is required to have optimal
        parenthesization for at lease one of its terms.  This requirement
        attempts to accelerate the constriction searching by having a smaller
        number of branches at the first-edge level of the recursion tree.
        However, it has a chance of giving deteriorated optimization, and it is
        not guaranteed to be faster since pivoting at this level have to be
        disabled.  So it is set as False by default.  It might be worth
        experimenting for large inputs, especially with exhaust strategy for
        contractions, or when greedy is turned on.

    greedy_cutoff
        The depth cutoff for making greedy selection in constriction. Beyond
        this depth in the recursion tree (inclusive), only the choices making
        locally best saving will be considered.  With negative values,
        full Bron-Kerbosch backtracking is performed.

    drop_cutoff
        The depth cutoff for picking only a random one with greedy saving in
        summation optimization.  The difference with the option
        ``greedy_cutoff`` is that here only **one** choice giving the locally
        best saving will be considered, rather than all of them.  This could
        give better acceleration than ``greedy_cutoff`` at the presence of large
        degeneracy, while results could be less optimized.  For large inputs, a
        value of ``2`` is advised.

    remove_shallow
        Shallow intermediates are outer-product intermediates that come with no
        summations.  Normally these intermediates cannot give saving big enough
        to justify their memory usage.  So by default, they just dropped, with
        their content inlined into places where they are referenced.

    """

    # This interface function is primarily just for sanity checking and
    # normalization of the input.

    substs = {} if substs is None else substs

    computs = [
        i.simplify() if simplify else i.reset_dumms()
        for i in computs
    ]
    if len(computs) == 0:
        raise ValueError('No computation is given!')

    if not isinstance(contr_strat, ContrStrat):
        raise TypeError('Invalid contraction strategy', contr_strat)

    opt = _Optimizer(
        computs, substs=substs, interm_fmt=interm_fmt,
        contr_strat=contr_strat, opt_sum=opt_sum,
        repeated_terms_strat=repeated_terms_strat,
        opt_symm=opt_symm, req_an_opt=req_an_opt,
        greedy_cutoff=greedy_cutoff, drop_cutoff=drop_cutoff,
        remove_shallow=remove_shallow
    )

    return opt.optimize()


#
# The internal optimization engine
# --------------------------------
#
# General small type definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Base for tensor definitions.
#
# Symbol for 0-order tensors, IndexedBase for other cases.

_Base = typing.Union[Symbol, IndexedBase]

# Symbol/range pairs.

_SrPairs = typing.Sequence[typing.Tuple[Symbol, Range]]

# Sequences of terms.

_Terms = typing.Sequence[Term]

# Indices to tensor bases.

_Indices = typing.Tuple[Expr]


class _Grain(typing.NamedTuple):
    """A piece of grain ready for optimization.

    Basically it is a tensor definition with localized terms.
    """
    base: _Base
    exts: _SrPairs
    terms: _Terms


class _IntermRef(typing.NamedTuple):
    """A reference to an intermediate."""
    coeff: Expr
    base: _Base
    indices: _Indices
    power: int

    @property
    def ref(self):
        """The reference to intermediate without coefficient."""
        return _index(self.base, self.indices) ** self.power


#
# Utility constants
# ~~~~~~~~~~~~~~~~~
#


_ZERO = Integer(0)
_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)

_EXT = 0
_SUMMED_EXT = 1
_SUMMED = 2

_SUBSTED_EVAL_BASE = Symbol('gristmillSubstitutedEvalBase')


#
# Global factories
# ~~~~~~~~~~~~~~~~
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


#
# Utility static functions
# ~~~~~~~~~~~~~~~~~~~~~~~~
#


def _get_canon_coeff(coeffs, preferred):
    """Get the canonical coefficient from a list of coefficients."""

    expr = sum(
        v * _SYMB_FACTORY[i] for i, v in enumerate(coeffs)
    ).together()

    frac = _UNITY  # The fractional part.
    if isinstance(expr, Mul):
        for i in expr.args:
            if isinstance(i, Pow) and i.args[1] < 0:
                frac *= i
            continue
        expr /= frac

    coeff, _ = primitive(expr, *[
        _SYMB_FACTORY[i] for i, _ in enumerate(coeffs)
    ])

    # Initial coefficient without phase.
    init_coeff = coeff * frac

    # The primitive computation does not take phase into account.
    negs = []
    poses = []
    for i in coeffs:
        i /= init_coeff
        if i.has(_NEG_UNITY) or i.is_negative:
            negs.append(-i)
        else:
            poses.append(i)
        continue

    neg_sig, pos_sig = [
        (len(i), tuple(sorted(default_sort_key(j) for j in i)))
        for i in [negs, poses]
    ]
    if neg_sig > pos_sig:
        phase = _NEG_UNITY
    elif pos_sig > neg_sig:
        phase = _UNITY
    else:
        preferred_phase = (
            _NEG_UNITY if preferred.has(_NEG_UNITY) or preferred.is_negative
            else _UNITY
        )
        phase = preferred_phase

    return (coeff * phase * frac).simplify()


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


class _Interm(typing.NamedTuple):
    """Newly formed intermediate.

    This small utility carries both a symbolic reference to an intermediate
    and the actual node for this, which can be helpful for getting
    information about a newly-formed intermediate.
    """
    ref: Expr
    node: _EvalNode


#
# Internals for product optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def _get_prod_final_cost(exts_total_size, sums_total_size) -> Size:
    """Compute the final cost for a pairwise product evaluation."""

    if sums_total_size == 1:
        return exts_total_size
    else:
        return 2 * exts_total_size * sums_total_size


def _gen_broken_sums(sum_chunks):
    """Generate broken summations in increasing size of broken summations.

    The size and the actual subset of broken summations are generated.
    """

    n_chunks = len(sum_chunks)

    # The entries in the heap queue are triples: the total size, the subset of
    # summation chunks, and the actual subset of summations.
    init = Tuple4Cmp((1, 0, 0))  # Nothing is broken.
    queue = [init]
    while len(queue) > 0:
        curr_size, curr_chunks, curr_broken = heapq.heappop(queue)
        yield curr_size, curr_broken
        next_idx = curr_chunks.bit_length()
        if next_idx < n_chunks:
            new_size, new_broken = sum_chunks[next_idx]
            joined_size = curr_size * new_size
            joined_chunks = curr_chunks | 1 << next_idx
            joined_broken = curr_broken | new_broken
            heapq.heappush(queue, Tuple4Cmp((
                joined_size, joined_chunks, joined_broken
            )))
            if next_idx > 0:
                top_idx = next_idx - 1
                top_size, top_broken = sum_chunks[top_idx]
                new_size, rem = divmod(joined_size, top_size)
                assert rem == 0
                assert joined_chunks & 1 << top_idx > 0
                assert joined_broken & top_broken == top_broken
                heapq.heappush(queue, Tuple4Cmp((
                    new_size, joined_chunks ^ 1 << top_idx,
                    joined_broken ^ top_broken
                )))
        continue


#
# Internals for summation optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Convention: nodes always refer to nodes in the DAG for tensor computations,
# while vertices are used for vertices in the constriction graph.
#

# Organized references to products in a summation.
#
# Intermediate base -> (indices -> coefficient)

_OrgTerms = typing.DefaultDict[
    Symbol, typing.DefaultDict[typing.Tuple[Expr, ...], Expr]
]

#
# Symbolic names for the parts of the bicliques.
#

_LEFT = 0
_RIGHT = 1
_OPPOS = {
    _LEFT: _RIGHT,
    _RIGHT: _LEFT
}

# For type annotation, actually is should be ``_LEFT | _RIGHT`` in Haskell
# algebraic data type notation.

_LR = typing.NewType('_LR', int)

_LRS = (_LEFT, _RIGHT)


class _LastStepIdxes(typing.NamedTuple):
    """The involved indices of the last step of a constriction.

    The external and summation indices involved by the left/right factor in
    the last step of a contraction.  This is going to be used as the key for
    accessing the actual graph.

    """
    exts: typing.Tuple[_SrPairs, _SrPairs]
    sums: _SrPairs


class _EdgeInfo(typing.NamedTuple):
    """Information about an edge on a constriction graph."""
    term: int
    eval_: _Prod
    coeff: Expr
    exc_cost: Size


class _BaseInfo:
    """Information about a base referenced in a sum node.

    This is an open struct, with most of its manipulation done inside the
    optimizer class.
    """

    __slots__ = [
        'count',
        'base',
        'node'
    ]

    def __init__(self, base: _Base, node: _Prod):
        """Initialize the information.

        The count is initialized to **zero**.
        """

        self.base = base
        self.node = node
        self.count: int = 0


#
# Intermediate data and results for the Kron-Kerbosch process.
#


class _VertInfo(typing.NamedTuple):
    """Information about a vertex in a constriction graph.

    expr
        The original expression for the factor.

    exts
        The involved external indices.

    canon
        The canonicalized content for the factor.

    """

    exts: int
    expr: Expr
    canon: _Terms


class _Delta(object):
    """Additional information about augmentation by a designated vertex.
    """

    __slots__ = [
        'coeff',
        'leading_coeff',
        'terms',
        'exc_cost',
        'saving'
    ]

    def __init__(
            self, coeff: Expr, leading_coeff: typing.Optional[Expr],
            terms: int, exc_cost: Size
    ):
        """Initialize the delta."""
        self.coeff = coeff
        self.leading_coeff = leading_coeff
        self.terms = terms
        self.exc_cost = exc_cost
        self.saving: Size = 0


class _DesVert(typing.NamedTuple):
    """Vertices designated for a specific part."""
    part: int
    vert: int


# Sets of designated vertices.

_DesVerts = typing.Set[_DesVert]

# Dictionary of the designated vertices augmenting the current biclique along
# with their delta.

_DesVertsWDelta = typing.Mapping[_DesVert, _Delta]

# Zipped vertices and coefficients.

_VertsWCoeff = typing.List[typing.Tuple[int, Expr]]

# Parts for a constriction, left and right.

_ConstrParts = typing.Tuple[_VertsWCoeff, _VertsWCoeff]


class _Biclique(typing.NamedTuple):
    """A biclique to be yielded."""
    parts: _ConstrParts
    leading_coeff: Expr
    terms: int
    saving: Size
    constr_graph: '_ConstrGraph'


#
# Cost-related utilities for the Kron-Kerbosch process.
#


class _CostCoeffs(typing.NamedTuple):
    """Cached quantities for getting gross saving of bicliques.

    final
        The final cost for contraction and make an addition of the results.

    preps
        The cost of making an addition for left and right factors.

    """

    final: Size
    preps: typing.Tuple[Size, Size]


def _get_cost_coeffs(last_step_idxes: _LastStepIdxes) -> _CostCoeffs:
    """Get the cost coefficients for the given last step indices."""

    sums = last_step_idxes.sums
    exts = last_step_idxes.exts

    ext_size = get_total_size(itertools.chain.from_iterable(exts))

    final = _get_prod_final_cost(
        ext_size, get_total_size(sums)
    ) + ext_size

    preps = (
        get_total_size(itertools.chain(exts[0], sums)),
        get_total_size(itertools.chain(exts[1], sums))
    )  # Explicitly repeated for linter.

    return _CostCoeffs(final=final, preps=preps)


class _VertGross(typing.Dict[typing.Tuple[int, int], typing.Tuple[Size, Size]]):
    """Gross saving of vertices.

    Given any numbers of vertices in the two parts, the gross saving of an
    additional vertex in the two parts can be queried.  The result are
    memorized for performance.

    """

    __slots__ = [
        '_cost_coeffs'
    ]

    def __init__(self, last_step_idxes: _LastStepIdxes):
        """Initialize the dictionary."""
        self._cost_coeffs = _get_cost_coeffs(last_step_idxes)

    def __missing__(self, key):
        """Compute the gross savings for new keys."""
        assert len(key) == 2
        assert all(i >= 0 for i in key)

        if any(i == 0 for i in key):
            res = (0, 0)
        else:
            cost_coeffs = self._cost_coeffs
            res = tuple(
                key[_OPPOS[i]] * cost_coeffs.final - cost_coeffs.preps[i]
                for i in _LRS
            )

        self[key] = res
        return res


#
# The core classes.
#


class _BronKerbosch:
    """Iterable for the maximal bicliques.

    For performance reasons, the bicliques generated will contain references
    to internal mutable data.  It is the **responsibility of the caller** to
    make proper copy when it is necessary.

    """

    def __init__(
            self, last_step_idxes: _LastStepIdxes, constr_graph: '_ConstrGraph'
    ):
        """Initialize the iterator."""

        # Static data during the recursion, cached here for easier and faster
        # access.
        self._constr_graph = constr_graph
        self._opt = constr_graph.constr_graphs.opt
        self._req_an_opt = self._opt.req_an_opt
        self._greedy_cutoff = self._opt.greedy_cutoff
        self._drop_cutoff = self._opt.drop_cutoff

        # Dynamic data during the recursion.
        #
        # Zipped nodes and coefficients, for left and right.
        self._curr: _ConstrParts = ([], [])

        # The leading coefficient.
        self._leading_coeff = None

        # The stack of actual saving.
        #
        # Keeping the saving as stack could save the cost of subtraction by
        # using some additional memory.
        self._savings = []

        # The set of terms currently in the biclique.
        self._terms = 0

        # Gross saving of new vertices.
        self._vert_gross: _VertGross = _VertGross(last_step_idxes)

    def __iter__(self):
        """Iterate over the maximal bicliques."""

        exts = self._constr_graph.exts
        # All left and right nodes.
        subg = {
            _DesVert(part=part, vert=vert): _Delta(
                coeff=_UNITY, leading_coeff=None, terms=0, exc_cost=0
            )
            for vert, info in self._constr_graph.verts
            for part in _LRS
            if info.exts == exts[part]
        }

        assert len(subg) > 0

        yield from self._expand(subg, set(subg.keys()))

        # If things all goes correctly, the stack should be reverted to initial
        # state by now.
        assert all(len(i) == 0 for i in self._curr)
        assert self._terms == 0
        assert len(self._savings) == 0
        assert self._leading_coeff is None

        return

    def _expand(
            self, subg: _DesVertsWDelta, cand: _DesVerts,
    ):
        """Generate the bicliques from the current state.

        This is the core of the Bron-Kerbosch algorithm.
        """

        # Cached variables of the current state.
        curr = self._curr
        n_verts = tuple(len(i) for i in curr)

        savings = self._savings
        depth = len(savings)
        curr_saving = savings[-1] if depth > 0 else 0
        exts = self._constr_graph.exts

        # The code here are adapted from the code in NetworkX for maximal clique
        # problem of simple general graphs.  The original code are kept as much
        # as possible and put in comments.  The original code on which the code
        # is based can be found at,
        #
        # https://github.com/networkx/networkx/blob
        # /48f4b5736174844c77044fae90e3e7adf1dabc10/networkx/algorithms
        # /clique.py#L277-L299
        #

        if_maximal = all(i.saving < 0 for i in subg.values())

        # Redundant check on biclique size is used to skip the possibly
        # expansive saving comparison.
        if_profitable = all(
            i > 0 for i in n_verts
        ) and any(i > 1 for i in n_verts) and curr_saving >= 0

        if if_maximal and if_profitable:
            # If maximal and profitable.
            #
            # if not subg_q:
            #    yield Q[:]
            #
            yield _Biclique(
                parts=curr, leading_coeff=self._leading_coeff,
                terms=self._terms, saving=curr_saving,
                constr_graph=self._constr_graph
            )

        # The quadratic loop.
        subgq = {}
        for q_v, q_d in subg.items():
            subg_q = {}
            subgq[q_v] = subg_q
            for r_v, r_d in subg.items():
                updated_r_d = self._update_delta(q_v, q_d, r_v, r_d)
                if updated_r_d is not None:
                    subg_q[r_v] = updated_r_d
                continue
            continue

        #
        # u = max(subg, key=lambda u: len(cand & adj[u]))
        # for q in cand - adj[u]:
        #

        # to_loop need to be eagerly evaluated for avoiding complication with
        # the mutation of cand during the loop and the set operations for
        # pivoting.

        pivots: typing.Iterable[_DesVert] = []
        if n_verts[0] == 0:
            to_loop = {i for i in cand if i.part == 0}
        elif n_verts[1] == 0:
            to_loop = {i for i in cand if i.part == 1}
            if exts[0] == exts[1]:
                # First part, first vertex, the vertex
                exist_vert: int = curr[0][0][0]
                to_loop = {i for i in to_loop if i.vert > exist_vert}
            if self._req_an_opt:
                to_loop = {i for i in to_loop if subg[i].exc_cost == 0}
            else:
                gross = self._vert_gross[(1, 1)][1]
                pivots = (
                    k for k, v in subg.items()
                    if k.part == 1 and gross - v.exc_cost >= 0
                )
        else:
            to_loop = {i for i in cand if subg[i].saving >= 0}
            if len(to_loop) == 0:
                return

            pivots = (k for k, v in subg.items() if v.saving > 0)

            cut_greedy = 0 <= self._greedy_cutoff <= depth
            cut_full = 0 <= self._drop_cutoff <= depth
            if cut_greedy or cut_full:
                greedy_saving = max(subg[i].saving for i in to_loop)
                to_loop = {
                    i for i in to_loop if subg[i].saving == greedy_saving
                }
                if cut_full:
                    to_loop = {to_loop.pop()}
                    pivots = []

        # Designated vertices that can be excluded for each pivot.
        fqs = (
            {i for i in subgq[k].keys() if i.part == k.part}
            for k in pivots
        )
        try:
            excl = max(fqs, key=lambda x: len(x & to_loop))
        except ValueError:
            pass
        else:
            to_loop -= excl

        for q_v in to_loop:
            q_d = subg[q_v]
            part, vert = q_v.part, q_v.vert

            #
            # cand.remove(q)
            #
            cand.remove(q_v)

            #
            # Q.append(q)
            #
            curr[part].append((vert, q_d.coeff))
            if q_d.leading_coeff is not None:
                self._leading_coeff = q_d.leading_coeff
            new_terms = q_d.terms
            assert self._terms & new_terms == 0
            self._terms |= new_terms
            savings.append(curr_saving + q_d.saving)

            #
            # adj_q = adj[q]
            # subg_q = subg & adj_q
            #
            subg_q = subgq[q_v]

            #
            # if not subg_q:
            #    yield Q[:]
            #
            # Moved to top for clarity.
            #

            #
            # cand_q = cand & adj_q
            #
            cand_q = {i for i in cand if i in subg_q}

            # if cand_q:
            #     for clique in expand(subg_q, cand_q):
            #         yield clique

            yield from self._expand(subg_q, cand_q)

            #
            # Q.pop()
            #
            curr[part].pop()
            assert self._terms & new_terms == new_terms
            self._terms ^= new_terms
            savings.pop()
            if q_d.leading_coeff is not None:
                self._leading_coeff = None

            continue

    def _update_delta(
            self, new_v: _DesVert, new_d: _Delta,
            curr_v: _DesVert, curr_d: _Delta
    ) -> typing.Optional[_Delta]:
        """Update the delta assuming a new node is added to the stack.

        This is the core and performance bottleneck of the Bron-Kerbosch
        algorithm.
        """

        new_terms = new_d.terms
        curr_terms = curr_d.terms
        if new_terms & curr_terms != 0:
            return None

        new_p = new_v.part
        curr_p = curr_v.part
        curr_coeff = curr_d.coeff
        new_leading_coeff = new_d.leading_coeff
        curr_leading_coeff = curr_d.leading_coeff

        updated_d = _Delta(
            coeff=curr_coeff, leading_coeff=curr_leading_coeff,
            terms=curr_terms, exc_cost=curr_d.exc_cost
        )

        if new_p == curr_p:
            if new_leading_coeff is not None:
                assert curr_leading_coeff is not None
                updated_d.coeff = (
                    curr_leading_coeff / new_leading_coeff
                ).simplify()
                updated_d.leading_coeff = None
        else:

            new_neighb = self._constr_graph.graph[new_v.vert]
            curr_vert = curr_v.vert
            if curr_vert not in new_neighb:
                return None
            edge = new_neighb[curr_vert]['info']

            edge_term = 1 << edge.term
            if_conflict = (
                edge_term & new_terms != 0 or edge_term & curr_terms != 0 or
                edge_term & self._terms != 0
            )
            if if_conflict:
                return None
            updated_d.terms |= edge_term

            updated_d.exc_cost += edge.exc_cost

            edge_coeff = edge.coeff

            if new_leading_coeff is not None:
                # The previous node gives the first edge.
                updated_d.coeff = (
                    edge_coeff / new_leading_coeff
                ).simplify()
            elif self._leading_coeff is None:
                # This node gives the first edge.
                updated_d.leading_coeff = edge_coeff
            else:
                proj = edge_coeff / (self._leading_coeff * new_d.coeff)
                if (proj - curr_coeff).simplify() != 0:
                    return None

        n_verts = [len(i) for i in self._curr]
        n_verts[new_p] += 1
        gross = self._vert_gross[tuple(n_verts)][curr_p]

        updated_d.saving = gross - updated_d.exc_cost

        return updated_d


class _ConstrGraph:
    """Constriction graph for a given involvement of indices.

    We have separate graphs for different involved indices combinations.  For
    each combination, the graph has the factors as vertices, and actual
    evaluations with the factors as edges.  Internally, the graph is stored
    as a NetworkX graph.

    """

    def __init__(
            self, constr_graphs: '_ConstrGraphs', exts_l: int, exts_r: int
    ):
        """Initialize the constriction graph.

        graphs
            The constriction graphs.

        exts_l, exts_r

            The pair of integers encoding the external indices involved by
            the two parts of the graph.

        """

        self.constr_graphs = constr_graphs
        self.exts = (exts_l, exts_r)

        self.graph = Graph()
        self._verts = {}  # From canonicalized factor to the vertex number.
        self.terms = 0

        # The optimal biclique in the current graph.  None when it is not yet
        # determined, False when it is determined that there is no profitable
        # biclique in the current graph.
        self._opt_saving = None
        self._opt_biclique = None

    @property
    def verts(self):
        """The nodes in the graph as integers with the information."""
        return (
            (i, j) for i, j in self.graph.nodes(data='info')
        )

    def add_edge(
            self, node_infos: typing.Tuple[_VertInfo, _VertInfo],
            coeff: Expr, term: int, eval_: _Prod
    ):
        """Add a new edge to the graph."""

        graph = self.graph
        term_bases = self.constr_graphs.term_bases
        repeated_terms_strat = self.constr_graphs.opt.repeated_terms_strat

        # Treat excess cost first, since it might lead to direct return.
        base_info = term_bases[term]
        count = base_info.count
        assert count > 0
        if count == 1 or repeated_terms_strat == RepeatedTermsStrat.IGNORE:
            exc_cost = eval_.total_cost - base_info.node.total_cost
        elif repeated_terms_strat == RepeatedTermsStrat.SKIP:
            return
        elif repeated_terms_strat == RepeatedTermsStrat.NATURAL:
            exc_cost = eval_.total_cost
        else:
            exc_cost = None  # For linter.
            assert None

        nodes = []
        for i in node_infos:
            canon = i.canon
            if canon in self._verts:
                idx = self._verts[canon]
            else:
                idx = len(self._verts)
                self._verts[canon] = idx
                graph.add_node(idx, info=i)
            nodes.append(idx)
            continue

        edge_info = _EdgeInfo(
            term=term, eval_=eval_, coeff=coeff, exc_cost=exc_cost
        )

        n1, n2 = nodes
        neighb1 = graph[n1]
        if n2 in neighb1:
            # It is possible that two evaluations actually the same be
            # recorded twice in the evaluation of product nodes because of
            # symmetry.
            assert neighb1[n2]['info'].term == edge_info.term
        else:
            graph.add_edge(*nodes, info=edge_info)

        self.terms |= 1 << term

    def get_opt_biclique(
            self, last_step_idxes: _LastStepIdxes
    ) -> typing.Tuple[typing.Optional[Size], typing.Optional[_Biclique]]:
        """Get the optimal biclique in the current graph.
        """

        if self._opt_saving is not None:
            if self._opt_saving is False:
                return None, None
            else:
                return self._opt_saving, self._opt_biclique

        opt_saving = None
        opt_biclique = None

        for biclique in _BronKerbosch(last_step_idxes, self):

            saving = biclique.saving

            if opt_saving is None or saving > opt_saving:
                opt_saving = saving
                # Make copy only when we need them.
                parts = biclique.parts
                assert len(parts) == 2
                opt_biclique = _Biclique(
                    parts=(list(parts[0]), list(parts[1])),
                    leading_coeff=biclique.leading_coeff,
                    terms=biclique.terms, saving=biclique.saving,
                    constr_graph=biclique.constr_graph
                )

            continue

        if opt_saving is None:
            assert opt_biclique is None
            self._opt_saving = False
            self._opt_biclique = None
            return None, None
        else:
            self._opt_saving = opt_saving
            self._opt_biclique = opt_biclique
            return opt_saving, opt_biclique

    def remove_terms(self, terms: int) -> bool:
        """Remove all edges for the given terms.

        Vertices no longer connected to anything is removed as well.  If a
        value of True is returned, we have an empty graph after the removal.
        """

        graph = self.graph

        if self.terms & terms != 0:
            edges2remove = [
                (n1, n2) for n1, n2, info in graph.edges(data='info')
                if 1 << info.term & terms != 0
            ]
            graph.remove_edges_from(edges2remove)
            nodes2remove = [
                i for i in graph.nodes() if graph.degree(i) == 0
            ]
            graph.remove_nodes_from(nodes2remove)

            self.terms ^= self.terms & terms

            # Reset cached optimal biclique.
            self._opt_saving = None
            self._opt_biclique = None

        return graph.number_of_nodes() == 0


class _ConstrGraphs(typing.Dict[_LastStepIdxes, _ConstrGraph]):
    """The constriction graphs from a sum of contractions.

    The constriction graphs are organized according to their external and
    summation indices involved by the factors in the last step to achieve
    better performance with one big graph separated into pieces. With this
    decomposition, for instance, we can cache maximum bicliques in subgraphs
    unaffected by the latest constriction.

    Here just the basic data is defined, with most actual operations directly
    performed inside the optimizer.

    Attributes
    ----------

    bases
        The mapping from the actual base to the base info.

    term_bases
        The list of base info for each of the terms.

    """

    __slots__ = [
        'opt',
        'bases',
        'term_bases'
    ]

    def __init__(self, opt: '_Optimizer'):
        """Initialize the graphs.

        Here only the most basic resource initialization is performed.
        """
        super().__init__()

        self.opt = opt

        self.bases: typing.Dict[_Base, _BaseInfo] = {}

        # None for plain scalar terms.
        self.term_bases: typing.List[typing.Optional[_BaseInfo]] = []

    def get_opt_biclique(self) -> typing.Tuple[
        typing.Optional[_LastStepIdxes], typing.Optional[_Biclique]
    ]:
        """Choose the most profitable biclique.
        """

        opt_saving = None
        opt_last_step_idxes = None
        opt_biclique = None
        for last_step_idxes, constr_graph in self.items():

            curr_opt_saving, curr_opt_biclique = constr_graph.get_opt_biclique(
                last_step_idxes
            )

            if curr_opt_saving is None:
                continue

            if opt_saving is None or curr_opt_saving > opt_saving:
                opt_saving = curr_opt_saving
                opt_last_step_idxes = last_step_idxes
                opt_biclique = curr_opt_biclique

            continue

        return opt_last_step_idxes, opt_biclique

    def cleanup_constred(self, if_untouched: int, biclique: _Biclique) -> int:
        """Clean up the terms after a constriction."""

        terms = biclique.terms
        assert if_untouched & terms == terms
        if_untouched ^= terms

        to_remove = []
        for last_step_idxes, constr_graph in self.items():
            if_empty = constr_graph.remove_terms(biclique.terms)
            if if_empty:
                to_remove.append(last_step_idxes)
            continue
        for i in to_remove:
            del self[i]
            continue

        return if_untouched


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

    def __init__(
            self, computs, substs, interm_fmt,
            contr_strat, opt_sum, repeated_terms_strat, opt_symm, req_an_opt,
            greedy_cutoff, drop_cutoff, remove_shallow
    ):
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

        # Storage of user options to be accessed during the optimization.
        #
        # Public for the each of accessing from other internal classes.
        self.interm_fmt = interm_fmt
        self.contr_strat = contr_strat
        self.opt_sum = opt_sum
        self.repeated_terms_strat = repeated_terms_strat
        self.opt_symm = opt_symm
        self.req_an_opt = req_an_opt
        self.greedy_cutoff = greedy_cutoff
        self.drop_cutoff = drop_cutoff
        self.remove_shallow = remove_shallow

        # Other internal data preparation.
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
            sums = self._proc_sums(term.sums, substs, sort=True)
            amp = term.amp

            # Add the true free symbols to the exclusion set.
            self._excl |= term.free_vars - ext_symbs
            terms.append(Term(sums, amp, ()))

            continue

        return _Grain(
            base=comput.base if len(exts) == 0 else comput.base.args[0],
            exts=exts, terms=terms
        )

    def _proc_sums(self, sums, substs, sort=False):
        """Process a summation list.

        The ranges will be replaced with substituted sizes.  Relevant members of
        the optimizer will also be updated.  User error will also be reported.
        """

        res = []
        for symb, range_ in sums:

            new_range, range_var = form_sized_range(range_, substs)

            if range_var is not None:
                if self._range_var is None:
                    self._range_var = range_var
                elif self._range_var != range_var:
                    raise ValueError(
                        'Invalid range', range_, 'unexpected symbol',
                        range_var, 'conflicting with', self._range_var
                    )
                else:
                    pass

            if new_range not in self._input_ranges:
                self._input_ranges[new_range] = range_
            elif range_.size != self._input_ranges[new_range].size:
                raise ValueError(
                    'Invalid ranges', (range_, self._input_ranges[new_range]),
                    'duplicated labels'
                )
            else:
                pass

            res.append((symb, new_range))
            continue

        if sort:
            res.sort(key=lambda x: x[1].size)

        return tuple(res)

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
        """

        if len(node.evals) == 0:
            self._optimize(node)

        # We need to find an evaluation with optimal cost.
        assert len(node.evals) > 0
        node.evals = [next(
            i for i in node.evals if i.total_cost == node.total_cost
        )]
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
        exts_dict = dict(node.exts)
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
                factors, term_coeff = term.get_amp_factors(
                    self._interms, exts_dict
                )

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

        # Cache some properties.
        remove_shallow = self.remove_shallow

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

                if_shallow = (
                    remove_shallow and len(terms) == 1
                    and len(terms[0].sums) == 0
                )
                if if_shallow:
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
                )(self.interm_fmt.format(next_idx))
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
        for idx, term in enumerate(terms):
            term, canon_sums = self._canon_term(new_sums, term)

            factors, coeff = term.get_amp_factors(self._interms)
            coeffs.append(coeff)

            candidates[
                term.map(lambda x: prod_(factors))
            ].append((canon_sums, idx))
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

        canon_new_sums = set(i for i, _ in chosen[1])
        if len(canon_new_sums) > 1:
            warnings.warn(
                'Internal deficiency: '
                'summation intermediate may not be fully canonicalized'
            )
        # This could also fail when the chosen term has symmetry among the new
        # summations not present in any other term.  This can be hard to check.

        canon_new_sum = canon_new_sums.pop()
        preferred = prod_(
            coeffs[i] for _, i in candidates[chosen[0]]
        )
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

        if len(terms) == 0:
            raise ValueError(
                'Tensor is constant zero, probably it is not what you meant',
                grain.base
            )
        return self._form_sum_from_terms(
            grain.base, exts, terms
        )

    def _optimize(self, node):
        """Optimize the evaluation of the given node.

        The evaluation methods will be filled with, possibly multiple, method of
        evaluations.
        """

        # For node with known evaluations, skip actual optimization.  This
        # enables the acceleration from dynamic programming.
        if len(node.evals) > 0:
            return node

        if isinstance(node, _Sum):
            return self._optimize_sum(node)
        elif isinstance(node, _Prod):
            return self._optimize_prod(node)
        else:
            assert False

    def _form_prod_interm(self, exts, sums, factors) -> _Interm:
        """Form a product intermediate.

        The factors are assumed to be all non-trivial factors needing
        processing.
        """

        decored_exts = tuple(
            (i, j.replace_label((j.label, _EXT)))
            for i, j in exts
        )
        n_exts = len(decored_exts)
        term = Term(tuple(sums), prod_(factors).simplify(), ())

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

            # The external symbols will automatically be considered in
            # get_amp_factors since they are in the summation list right now.
            key_factors, key_coeff = key_term.get_amp_factors(self._interms)
            interm = _Prod(
                base, key_exts, key_sums, key_coeff, key_factors
            )
            self._interms[base] = interm

        return _Interm(
            ref=coeff * _index(base, canon_exts, strip=True),
            node=self._interms[base]
        )

    def _form_sum_interm(
            self, exts: _SrPairs, terms: typing.Sequence[Term]
    ) -> _Interm:
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

        return _Interm(
            ref=coeff * _index(base, canon_exts, strip=True),
            node=self._interms[base]
        )

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
            factors, coeff = term.get_amp_factors(self._interms, ext_symbs)
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
        scalars, terms, _ = self._organize_sum_terms(sum_node.sum_terms)

        if self.opt_sum:
            new_terms, old_terms = self.constr_sum(terms, exts)
        else:
            new_terms = []
            old_terms = terms

        if self.opt_symm:
            old_terms = self._optimize_common_symmtrization(old_terms, exts)

        res_terms = scalars + old_terms + new_terms
        sum_node.evals = [_Sum(
            sum_node.base, sum_node.exts, res_terms
        )]
        return

    def _organize_sum_terms(self, terms: typing.Iterable[Expr]) -> typing.Tuple[
        typing.List[Expr], typing.List[Expr], _OrgTerms
    ]:
        """Organize terms in the summation node.
        """

        # Intermediate base -> (indices -> coefficient)
        #
        # This first gather terms with the same reference to deeper nodes.
        org_terms = collections.defaultdict(
            lambda: collections.defaultdict(lambda: _ZERO)
        )

        plain_scalars = []
        for term in terms:
            ref = self._parse_interm_ref(term)
            if ref is None:
                plain_scalars.append(term)
                continue
            assert ref.power == 1

            org_terms[ref.base][ref.indices] += ref.coeff
            continue

        res_terms = []

        for k, v in org_terms.items():
            assert len(v) > 0

            for indices, coeff in v.items():
                coeff = coeff.simplify()
                if coeff != 0:
                    res_terms.append(
                        _index(k, indices) * coeff
                    )

            continue

        return plain_scalars, res_terms, org_terms

    def _optimize_common_symmtrization(self, terms, exts):
        """Optimize common symmetrization in the intermediate references.
        """

        res_terms = []
        exts_dict = dict(exts)
        scalars, _, org_terms = self._organize_sum_terms(terms)
        assert len(scalars) == 0

        # Indices, coeffs tuple -> base, coeff
        pull_info = collections.defaultdict(list)
        for k, v in org_terms.items():

            if len(v) == 0:
                assert False
            elif len(v) == 1:
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

        return res_terms

    def constr_sum(
            self, terms: typing.Sequence[Expr], exts: _SrPairs
    ):
        """Constrict the summations greedily.
        """

        if_untouched = (1 << len(terms)) - 1
        new_terms = []

        constr_graphs = self._form_constr_graphs(terms, exts)

        while True:

            last_step_idxes, biclique = constr_graphs.get_opt_biclique()
            if last_step_idxes is None:
                break

            new_terms.append(self._form_constred_term(
                last_step_idxes, biclique
            ))
            if_untouched = constr_graphs.cleanup_constred(
                if_untouched, biclique
            )

            continue
        # End Main loop.

        untouched_terms = [
            v for i, v in enumerate(terms) if if_untouched & (1 << i) != 0
        ]
        return new_terms, untouched_terms

    def _form_constr_graphs(
            self, terms: typing.Sequence[Expr], exts: _SrPairs
    ) -> _ConstrGraphs:
        """Form the constriction graphs for the terms.

        The additional information about the bases of each of the terms are
        also returned.
        """

        constr_graphs = _ConstrGraphs(self)
        base_infos = constr_graphs.bases
        term_bases = constr_graphs.term_bases
        term_ref_nodes = []

        for term_idx, term in enumerate(terms):
            ref = self._parse_interm_ref(term)
            if ref is None:
                term_bases.append(None)
                term_ref_nodes.append((None, None))
                continue

            base = ref.base
            node = self._interms[base]
            assert isinstance(node, _Prod)
            term_ref_nodes.append((ref, node))

            if base in base_infos:
                base_info = base_infos[base]
            else:
                base_info = _BaseInfo(base, node)
                base_infos[base] = base_info

            base_info.count += 1
            term_bases.append(base_info)

            self._optimize(node)
            continue

        # This loop should have the correct bases count.
        for term_idx, term in enumerate(terms):
            ref, node = term_ref_nodes[term_idx]
            if ref is None:
                continue
            for eval_ in node.evals:
                assert isinstance(eval_, _Prod)
                self._aug_constr_graphs_4_eval(
                    constr_graphs, term_idx, ref, eval_, exts
                )
                continue

        return constr_graphs

    def _aug_constr_graphs_4_eval(
            self, res: _ConstrGraphs, term_idx: int, ref: _IntermRef,
            eval_: _Prod, exts: _SrPairs
    ):
        """Augment the constriction graphs for an evaluation.
        """

        if len(eval_.factors) < 2:
            return
        assert len(eval_.factors) == 2

        eval_terms = self._index_prod(eval_, ref.indices)
        assert len(eval_terms) == 1
        eval_term = eval_terms[0]

        ext_symbs = {i for i, _ in eval_.exts}

        factors, coeff = eval_term.get_amp_factors(
            self._interms, ext_symbs
        )
        coeff *= ref.coeff
        assert len(factors) == 2
        assert factors[0] != factors[1]

        sums = tuple(sorted(
            eval_term.sums, key=lambda x: (x[1], default_sort_key(x[0]))
        ))

        excl = set(self._excl)
        excl.update(ext_symbs)
        symms = self._drudge.symms.value

        factor_infos = []

        for f_i in factors:
            content = self._get_content(f_i)
            assert len(content) == 1
            content = content[0]

            symbs = f_i.atoms(Symbol)
            exts_idxes = tuple(
                i for i, v in enumerate(exts) if v[0] in symbs
            )
            exts_int = functools.reduce(operator.or_, (
                1 << i for i in exts_idxes
            ), 0)

            for i, _ in sums:
                assert i in symbs

            # In order to really make sure, the content will be re-canonicalized
            # based on the current ambient.
            canon = content.canon(symms=symms).reset_dumms(
                self._dumms, excl=excl | content.free_vars
            )[0]

            _, canon_coeff = canon.get_amp_factors(
                self._interms, ext_symbs
            )
            canon = canon.map(
                lambda x: x / canon_coeff, skip_vecs=True
            )
            coeff *= canon_coeff

            factor_infos.append((
                tuple(exts[i] for i in exts_idxes),
                _VertInfo(exts=exts_int, expr=f_i, canon=canon)
            ))
            continue

        factor_infos.sort(key=lambda x: x[1].exts)
        assert len(factor_infos) == 2
        last_step_idxes = _LastStepIdxes(
            exts=(factor_infos[0][0], factor_infos[1][0]), sums=sums
        )

        if last_step_idxes in res:
            constr_graph = res[last_step_idxes]
        else:
            constr_graph = _ConstrGraph(
                res, factor_infos[0][1].exts, factor_infos[1][1].exts
            )
            res[last_step_idxes] = constr_graph

        constr_graph.add_edge(
            (factor_infos[0][1], factor_infos[1][1]),
            coeff=coeff, term=term_idx, eval_=eval_
        )

        return

    def _form_constred_term(
            self, last_step_idxes: _LastStepIdxes, biclique: _Biclique
    ) -> Expr:
        """Form the factored term for the given constriction."""

        verts = biclique.constr_graph.graph.nodes

        # Form and optimize the two new summation nodes.
        factors = [biclique.leading_coeff]
        for exts_i, part_i in zip(last_step_idxes.exts, biclique.parts):
            scaled_terms = [
                verts[i]['info'].canon.scale(j) for i, j in part_i
            ]

            exts = tuple(itertools.chain(exts_i, last_step_idxes.sums))

            if len(scaled_terms) > 1:
                expr, eval_node = self._form_sum_interm(exts, scaled_terms)
            else:
                scaled_term = scaled_terms[0]
                expr, eval_node = self._form_prod_interm(
                    exts, scaled_term.sums, [scaled_term.amp]
                )
            factors.append(expr)
            self._optimize(eval_node)
            continue

        # Form the contraction node for the two new summation nodes.
        exts = tuple(sorted(
            set(itertools.chain.from_iterable(last_step_idxes.exts)),
            key=lambda x: default_sort_key(x[0])
        ))
        expr, eval_node = self._form_prod_interm(
            exts, last_step_idxes.sums, factors
        )

        # Make phony optimization of the intermediate.
        eval_node.total_cost = 1
        eval_node.evals = [eval_node]

        return expr

    #
    # Product optimization.
    #

    def _optimize_prod(self, prod_node):
        """Optimize the product evaluation node.
        """

        # This function should not be called on an already-optimized node.
        assert len(prod_node.evals) == 0

        n_factors = len(prod_node.factors)

        if n_factors < 2:
            assert n_factors == 1
            prod_node.evals.append(prod_node)
            sums_size = get_total_size(prod_node.sums)
            prod_node.total_cost = (
                get_total_size(prod_node.exts) * sums_size
            ) if sums_size != 1 else 0
            return

        contr_strat = self.contr_strat
        greedy_mode = contr_strat == ContrStrat.GREEDY
        normal_mode = (
            contr_strat == ContrStrat.OPT or contr_strat == ContrStrat.TRAV
        )
        exhaust_mode = contr_strat == ContrStrat.EXHAUST
        if_inclusive = (
            contr_strat == ContrStrat.TRAV or
            contr_strat == ContrStrat.EXHAUST
        )

        evals = prod_node.evals
        optimal_cost = None
        for final_cost, broken_sums, biparts_gen in self._gen_factor_biparts(
                prod_node
        ):
            def need_break() -> bool:
                """If we need to break the current loop."""
                if optimal_cost is None:
                    return False

                if greedy_mode:
                    return True
                elif normal_mode:
                    return final_cost > optimal_cost
                elif exhaust_mode:
                    return False
                else:
                    assert False

            if need_break():
                break
            # Else

            for bipart in biparts_gen:

                if need_break():
                    break

                # Recurse, two parts.
                assert len(bipart) == 2
                for i in bipart:
                    self._optimize(i.node)
                    continue

                total_cost = (
                    final_cost
                    + bipart[0].node.total_cost
                    + bipart[1].node.total_cost
                )

                if_new_optimal = (
                    optimal_cost is None or optimal_cost > total_cost
                )
                if if_new_optimal:
                    optimal_cost = total_cost
                    if not if_inclusive:
                        evals.clear()

                if if_new_optimal or if_inclusive:
                    new_eval = self._form_prod_eval(
                        prod_node, broken_sums, bipart
                    )
                    new_eval.total_cost = total_cost
                    evals.append(new_eval)

                continue

        assert len(evals) > 0
        prod_node.total_cost = optimal_cost
        return

    def _gen_factor_biparts(self, prod_node: _Prod):
        """Generate all the bipartitions of factors in a product node."""

        #
        # Compute things invariant to different summations for performance.
        #

        exts = prod_node.exts
        exts_total_size = get_total_size(exts)
        factors = prod_node.factors
        sums = prod_node.sums

        ext2idx, sum2idx = tuple(
            {v[0]: j for j, v in enumerate(i)}
            for i in (prod_node.exts, sums)
        )

        # Factors involving each of the summations, as iterable lists.
        sum_infos: typing.List[typing.List[int]] = [
            [] for _ in range(len(sum2idx))
        ]

        # Ext and sum involvements of factors.
        factor_infos = [[0, 0] for _ in factors]

        for i, v in enumerate(factors):
            for j in v.atoms(Symbol):
                if j in sum2idx:
                    sum_idx = sum2idx[j]
                    sum_infos[sum_idx].append(i)
                    factor_infos[i][1] |= 1 << sum_idx
                elif j in ext2idx:
                    factor_infos[i][0] |= 1 << ext2idx[j]
                else:
                    pass

        # Organize the summations: summations with exactly the same factor
        # involvement will be treated as a single chunk of summations.
        invol_sums = collections.defaultdict(list)
        for i, v in enumerate(sum_infos):
            invol_sums[tuple(v)].append(i)
            continue

        # Size and summation subsets.
        sum_chunks = []
        for v in invol_sums.values():
            sums_in_chunk = 0
            total_size = 1
            for i in v:
                sums_in_chunk |= 1 << i
                total_size *= sums[i][1].size
                continue
            sum_chunks.append((total_size, sums_in_chunk))
            continue
        sum_chunks.sort()

        #
        # Actual two-level generation.
        #

        for broken_size, broken in _gen_broken_sums(sum_chunks):
            broken_sums = [
                v for i, v in enumerate(sums) if broken & (1 << i)
            ]  # Sums to be retained in the evaluation.
            final_cost = _get_prod_final_cost(
                exts_total_size, broken_size
            )
            yield final_cost, broken_sums, self._gen_biparts_w_kept_sums(
                prod_node, broken, sum_infos, factor_infos
            )
            continue

    def _gen_biparts_w_kept_sums(
            self, prod_node: _Prod, broken, sum_infos, factor_infos
    ):
        """Generate all bipartitions with given summations kept.

        First the factors are divided into chunks indivisible according to
        the kept summations.  Then their bipartitions which really break the
        broken sums are generated.
        """

        n_factors = len(factor_infos)

        dsf = DSF(n_factors)

        for i, v in enumerate(sum_infos):
            if not (broken & 1 << i):
                dsf.union(v)
            continue

        if dsf.n_sets < 2:
            return

        # The sums, externals, and factors involved by each chunks.
        sums = []
        factors = []
        exts = []
        # Map root factors to the indices of the chunk in the above lists.
        indices = {}
        index = 0

        for i in dsf:
            root = dsf.find(i)
            if root not in indices:
                indices[root] = index
                index += 1
                factors.append(0)
                sums.append(0)
                exts.append(0)

            chunk = indices[root]
            factors[chunk] |= 1 << i
            exts[chunk] |= factor_infos[i][0]
            sums[chunk] |= factor_infos[i][1]

            continue

        # Loop over bipartitions of the indivisible chunks.

        n_chunks = index
        for p1 in range(1, 2 ** n_chunks - 1, 2):

            # Get the sums in the two chunks first.
            sums1, sums2 = 0, 0
            for i in range(n_chunks):
                if p1 & 1 << i:
                    sums1 |= sums[i]
                else:
                    sums2 |= sums[i]
                continue

            if all(i & broken == broken for i in (sums1, sums2)):

                # Only now we get the factors and the externals.
                factors1, factors2 = 0, 0
                exts1, exts2 = 0, 0
                for i in range(n_chunks):
                    if p1 & 1 << i:
                        factors1 |= factors[i]
                        exts1 |= exts[i]
                    else:
                        factors2 |= factors[i]
                        exts2 |= exts[i]
                    continue

                yield tuple(
                    self._form_part_interm(prod_node, broken, *i) for i in [
                        (exts1, sums1, factors1), (exts2, sums2, factors2)
                    ]
                )

        return

    def _form_part_interm(self, prod_node, broken, exts, sums, factors):
        """Form an intermediate for a partition for the given factors."""

        factors_list = [
            v for i, v in enumerate(prod_node.factors) if factors & 1 << i
        ]
        exts_list = [
            v for i, v in enumerate(prod_node.exts) if exts & 1 << i
        ]
        sums_list = []

        for i, v in enumerate(prod_node.sums):
            mask = 1 << i
            if not (sums & mask):
                # Sums not involved.
                continue
            elif broken & mask:
                exts_list.append(v)
            else:
                sums_list.append(v)
            continue

        return self._form_prod_interm(exts_list, sums_list, factors_list)

    def _form_prod_eval(
            self, prod_node: _Prod, broken_sums,
            parts: typing.Tuple[_Interm, ...]
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

        assert len(factors) == 2
        if factors[0] == factors[1]:
            factors = [factors[0] ** 2]
        return _Prod(
            prod_node.base, prod_node.exts, broken_sums,
            coeff * prod_node.coeff, factors
        )


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
