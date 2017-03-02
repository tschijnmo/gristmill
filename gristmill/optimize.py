"""Optimizer for the contraction computations."""

import collections
import itertools
import typing
import warnings

from drudge import TensorDef, prod_, Term, Range
from sympy import Integer, Symbol, Expr, IndexedBase, Mul, Indexed

from .utils import get_cost_key, add_costs


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
        self._drudge = None
        self._range_var = None
        self._excl = set()
        self._input_ranges = {}

        # Form pre-grist, basically everything is set except the dummy variables
        # for external indices and summations.
        pre_grist = [
            self._form_pre_grist(comput, substs) for comput in computs
            ]

        # Finalize grist formation by resetting the dummies.
        self._dumms = {
            k: self._drudge.dumms.value[v]
            for k, v in self._input_ranges.items()
            }

        self._grist = [self._reset_dumms(grain) for grain in pre_grist]

        return

    def _form_pre_grist(self, comput, substs):
        """Form grist from a given computation.
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
        for term in comput.local_terms:
            if not term.is_scalar:
                raise ValueError('Invalid term', term, 'expecting scalar')
            sums = self._proc_sums(term.sums, substs)
            amp = term.amp

            # Add the true free symbols to the exclusion set.
            self._excl |= term.free_vars - ext_symbs
            terms.append(Term(sums, amp, ()))

            continue

        return _Grain(exts=exts, terms=terms)

    def _proc_sums(self, sums, substs):
        """Process a summation list.

        The ranges will be replaced with substitution sizes.  Relevant members
        of the group will also be updated.  User error will also be reported.
        """

        res = []
        for symb, range_ in sums:

            if not range_.bounded:
                raise ValueError(
                    'Invalid range for optimization', range_,
                    'expecting explicit bound'
                )
            lower, upper = [
                self._check_range_var(range_, i.xreplace(substs))
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

    def _check_range_var(self, range_, expr) -> Expr:
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

    def _reset_dumms(self, grain):
        """Reset the dummies in a grain."""

        exts, ext_substs, dummbegs = Term.reset_sums(
            grain.exts, self._dumms, excl=self._excl
        )
        terms = []
        for term in grain.terms:
            sums, curr_substs, _ = Term.reset_sums(
                term.sums, self._dumms,
                dummbegs=dict(dummbegs), excl=self._excl
            )
            curr_substs.update(ext_substs)
            terms.append(term.map(
                lambda x: x.xreplace(curr_substs), sums=sums
            ))
            continue

        return _Grain(exts=exts, terms=terms)

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

    @staticmethod
    def _write_in_orig_ranges(sums):
        """Write the summations in terms of undecorated bare ranges."""
        return tuple(
            (i, j.replace_label(j.label[-1])) for i, j in sums
        )

    def _canon_terms(self, new_sums, terms):
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
        coeff_cnt = collections.Counter()

        candidates = collections.defaultdict(list)
        for term in terms:
            term, canon_sums = self._canon_term(new_sums, term)

            factors, coeff = term.amp_factors
            coeff_cnt[coeff] += 1

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
        canon_coeff = coeff_cnt.most_common(1)[0][0]
        res_terms = []
        for term in terms:
            term, _ = self._canon_term(canon_new_sum, term, fix_new=True)
            # TODO: Add support for complex conjugation.
            res_terms.append(term.map(lambda x: x / canon_coeff))
            continue

        return canon_coeff, tuple(
            sorted(res_terms, key=lambda x: x.sort_key())
        ), canon_new_sum

    def _canon_term(self, new_sums, term, fix_new=False):
        """Canonicalize a single term.

        Internal method for _canon_terms, not supposed to be directly called.
        """

        n_new = len(new_sums)
        term = Term(tuple(itertools.chain(
            (
                (v[0], v[1].replace_label((_EXT, i, v[1].label[-1])))
                for i, v in enumerate(new_sums)
            ) if fix_new else new_sums,
            (
                (i, j.replace_label((_SUMMED, j)))
                for i, j in term.sums
            )
        )), term.amp, ())
        canoned = term.canon(symms=self._drudge.symms.value)

        canon_sums = canoned.sums
        canon_orig_sums = self._write_in_orig_ranges(canon_sums)

        dumm_reset, _ = canoned.map(
            lambda x: x, sums=canon_orig_sums
        ).reset_dumms(
            dumms=self._dumms, excl=self._excl
        )

        canon_new_sums = canon_sums[:len(new_sums)]
        return dumm_reset.map(lambda x: x, sums=tuple(itertools.chain(
            (
                (i[0], j[1])
                for i, j in zip(dumm_reset.sums, canon_new_sums)
            ),
            dumm_reset.sums[n_new:]
        ))), canon_new_sums

    def _parse_interm_ref(self, sum_term: Expr):
        """Get the coefficient and pure intermediate reference in a reference.

        Despite being SymPy expressions, actually intermediate reference, for
        instance in a term in an summation node is very rigid.
        """

        if isinstance(sum_term, Mul):
            args = sum_term.args
            assert len(args) == 2
            if self._is_interm_ref(args[1]):
                return args
            else:
                assert self._is_interm_ref(args[0])
                return args[1], args[0]
        else:
            return _UNITY, sum_term

    def _is_interm_ref(self, expr: Expr):
        """Test if an expression is a reference to an intermediate."""
        return (isinstance(expr, Indexed) and expr.base in self._interms) or (
            expr in self._interms
        )

    #
    # General optimization.
    #

    def _form_node(self, grain: _Grain):
        """Form an evaluation node from a tensor definition.
        """

        # We assume it is fully simplified and expanded by grist preparation.
        exts = grain.exts
        terms = grain.terms

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

        The factors are assumed to be all non-trivial factors needing
        processing.
        """

        decored_exts = tuple(
            (i, j.replace_label((_EXT, j.label)))
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
            base = self._get_next_internal(len(exts) == 0)
            self._interms_canon[key] = base

            key_term = key[0]
            canon_exts = self._write_in_orig_ranges(key_term.sums[:n_exts])
            canon_sums = key_term.sums[n_exts:]
            canon_factors, canon_coeff = key_term.amp_factors
            interm = _Prod(
                canon_exts, canon_sums, canon_coeff, canon_factors
            )
            interm.canon = key
            interm.base = base
            self._interms[base] = interm

        return coeff * base[tuple(
            canon_exts[i][0] for i in range(n_exts)
        )]

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
