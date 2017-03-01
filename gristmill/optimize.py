"""Optimizer for the contraction computations."""

import collections
import typing

from drudge import TensorDef, prod_, Term, Range
from sympy import Integer, Symbol, Expr, IndexedBase, Number, Mul

from .utils import get_cost_key


#
#  The public driver
#  -----------------
#


def optimize(
        computs: typing.Iterable[TensorDef], substs=None, interm_fmt='tau{}',
        simplify=True
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

    """

    if simplify:
        computs = [i.simplify() for i in computs]
    else:
        computs = list(computs)

    opt = _Optimizer(
        computs, substs=substs, interm_fmt=interm_fmt
    )

    return opt.optimize()


#
# The internal optimization engine
# --------------------------------
#


class _Optimizer:
    """Optimizer for tensor contraction computations.

    This internal optimizer can only be used once for one set of input.
    """

    def __init__(self, computs, substs, interm_fmt):
        """Initialize the optimizer."""

        self._prepare_grist(computs, substs)

        self._interm_fmt = interm_fmt

        self._next_internal_idx = 0

        self._interms = {}
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
    # User input pre-processing and post-processing.
    #

    def _prepare_grist(self, computs, substs):
        """Prepare tensor definitions for optimization.
        """

        self._grist = []
        self._excl = set()
        self._input_ranges = {}
        pass

    def _linearize(self, optimized) -> typing.List[TensorDef]:
        """Linearize optimized forms of the evaluation.
        """
        pass

    #
    # Internal support utilities.
    #

    def _get_next_internal(self, symbol=False):
        """Get the base or symbol for the next internal intermediate.
        """
        idx = self._next_internal_idx
        self._next_internal_idx += 1
        cls = Symbol if symbol else IndexedBase
        return cls('gristmillInternalIntermediate{}'.format(idx))

    #
    # General optimization.
    #

    def _form_node(self, grain: TensorDef):
        """Form an evaluation node from a tensor definition.
        """

        # We assume it is fully simplified and expanded by grist preparation.
        terms = grain.rhs_terms
        exts = grain.exts

        if len(terms) == 0:
            assert False  # Should be removed by grist preparation.
        elif len(terms) == 1:
            term = terms[0]
            factors, coeff = term.amp_factors
            return _Prod(self, exts, term.sums, coeff, factors)
        else:

            sum_terms = []
            for term in terms:
                sums = term.sums
                factors, coeff = term.amp_factors
                interm_ref = self._form_prod_interm(exts, sums, factors)
                sum_terms.append(interm_ref * coeff)
                continue

            return _Sum(exts, sum_terms)

    def _optimize(self, node):
        """Optimize the evaluation of the given node."""

        if isinstance(node, _Sum):
            return self._optimize_sum(node)
        elif isinstance(node, _Prod):
            return self._optimize_prod(node)
        else:
            raise TypeError('Invalid node to optimize', node)

    def _form_prod_interm(self, exts, sums, factors) -> Expr:
        """Form a product intermediate.
        """
        return None

    def _form_sum_interm(self, collect_res) -> Expr:
        """Form a sum intermediate.
        """
        return None

    #
    # Sum optimization.
    #

    def _optimize_sum(self, sum_node: _Sum):
        """Optimize the summation node."""

        terms = sum_node.sum_terms

        while True:
            collectible, term_idxes = self._find_collectible(terms)
            if collectible is None:
                break
            terms = self._collect(terms, collectible, term_idxes)
            continue

        return _Sum(sum_node.exts, terms)

    def _find_collectible(self, terms):
        """Find a collectible factor among the terms in a sum."""

        collectibles = {}

        for idx, term in enumerate(terms):
            for i, j in self._get_collectibles(term):
                if i in collectibles:
                    collectibles[i].term_idxes.append(idx)
                else:
                    collectibles[i] = _CollectibleInfo(
                        saving_factor=j, term_idxes=[idx]
                    )
                continue
            continue

        return self._choose_collectible(collectibles)

    def _get_collectibles(self, term):
        """Get the collectible factors in a sum term."""
        pass

    def _choose_collectible(self, collectibles):
        """Choose the most profitable collectible factor."""
        chosen = max((
            (i, j.saving_factor * (len(j.term_idxes) - 1), j.term_idxes)
            for i, j in collectibles.items()
        ), key=lambda x: get_cost_key(x[1]))

        return (chosen[0], chosen[2]) if chosen[1] != 0 else (None, None)

    def _collect(self, terms, collectible, term_idxes):
        """Collect the given collectible factor."""

        new_terms = [
            v for i, v in enumerate(terms)
            if i not in set(term_idxes)
            ]  # Start with the untouched ones.

        to_collect = [
            self._collect_term(terms[i], collectible) for i in term_idxes
            ]
        new_ref = self._form_sum_interm(to_collect)

        new_terms.append(
            self._form_collected(collectible, new_ref)
        )

        return new_terms

    def _collect_term(self, term, collectible):
        """Collect the given collectible from the given term.
        """
        pass

    def _form_collected(self, collectible, new_ref):
        """Form new sum term with some factors collected.
        """
        pass

    #
    # Product optimization.
    #

    def _optimize_prod(self, prod_node):
        """Optimize the product evaluation node."""


class _EvalNode:
    """A node in the evaluation graph.
    """

    def __init__(self, exts):
        """Initialize the evaluation node.
        """

        self.exts = exts
        self.n_refs = 0


_CollectibleInfo = collections.namedtuple('_CollectibleInfo', [
    'saving_factor',
    'term_idxes'
])


class _Sum(_EvalNode):
    """Sum nodes in the evaluation graph."""

    def __init__(self, exts, sum_terms):
        """Initialize the node."""
        super().__init__(exts)
        self.sum_terms = sum_terms


class _Prod(_EvalNode):
    """Product nodes in the evaluation graph.
    """

    def __init__(self, exts, sums, coeff, factors, evals=None):
        """Initialize the node."""
        super().__init__(exts)
        self.sums = sums
        self.coeff = coeff
        self.factors = factors
        self.evals = evals
