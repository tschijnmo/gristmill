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

    def _get_def(self, interm_ref: Expr) -> typing.List[Term]:
        """Get the definition of an intermediate reference.

        The intermediate reference need to be a pure intermediate reference
        without any factor.
        """

        if isinstance(interm_ref, Indexed):
            base = interm_ref.base
            indices = interm_ref.indices
        elif isinstance(interm_ref, Symbol):
            base = interm_ref
            indices = None
        else:
            raise TypeError('Invalid intermediate reference', interm_ref)

        if base not in self._interms:
            raise ValueError('Invalid intermediate base', base)

        node = self._interms[base]
        substs = {}
        new_symbs = set()

        assert len(indices) == len(node.exts)
        for i, j in zip(indices, node.exts):
            substs[j[0]] = i
            new_symbs |= i.atoms(Symbol)
            continue

        res = []

        if isinstance(node, _Prod):

            # TODO: Add handling of sum intermediate reference in factors.
            term = Term(
                node.sums, node.coeff * prod_(node.factors), ()
            ).reset_dumms(
                self._dumms, excl=self._excl | new_symbs
            ).map(lambda x: x.xreplace(substs))
            res.append(term)

        elif isinstance(node, _Sum):
            for i in node.sum_terms:
                coeff, ref = self._parse_interm_ref(i.xreplace(substs))
                term = self._get_def(ref)[0].scale(coeff)
                res.append(term)
        else:
            assert False

        return res

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
            return _Prod(exts, term.sums, coeff, factors)
        else:
            return self._form_sum_from_terms(exts, terms)

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
            base = self._get_next_internal(n_exts == 0)
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
            i for i, _ in canon_exts
        )]

    def _form_sum_interm(self, exts, terms) -> Expr:
        """Form a sum intermediate.
        """

        decored_exts = [
            (i, j.replace_label((_EXT, j.label)))
            for i, j in exts
            ]
        n_exts = len(decored_exts)

        coeff, canon_terms, canon_exts = self._canon_terms(decored_exts, terms)

        if canon_terms in self._interms_canon:
            base = self._interms_canon[canon_terms]
        else:
            base = self._get_next_internal(n_exts == 0)

            node_exts = None
            node_terms = []
            for term in canon_terms:
                term_exts = self._write_in_orig_ranges(term.sums[:n_exts])
                if node_exts is None:
                    node_exts = term_exts
                else:
                    assert node_exts == term_exts
                node_terms.append(term.map(
                    lambda x: x, sums=term.sums[n_exts:]
                ))
                continue

            node = self._form_sum_from_terms(node_exts, node_terms)
            self._interms[base] = node

        return coeff * base[tuple(
            i for i, _ in canon_exts
        )]

    def _form_sum_from_terms(self, exts, terms):
        """Form a summation node for given the terms.

        No processing is done in this method.
        """
        sum_terms = []
        for term in terms:
            sums = term.sums
            factors, coeff = term.amp_factors
            interm_ref = self._form_prod_interm(exts, sums, factors)
            sum_terms.append(interm_ref * coeff)
            continue

        return _Sum(exts, sum_terms)

    #
    # Sum optimization.
    #

    def _optimize_sum(self, sum_node: _Sum):
        """Optimize the summation node."""

        # In this function, term is short for sum term.
        terms = list(sum_node.sum_terms)
        exts = sum_node.exts

        collectibles = collections.defaultdict(dict)  # type: _Collectibles

        new_term_idxes = list(i for i, _ in enumerate(terms))
        while True:

            for idx in new_term_idxes:
                term = terms[idx]

                for i, j in self._find_collectibles(exts, term):
                    infos = collectibles[i]
                    if idx not in infos:
                        # The same term cannot provide the same collectible
                        # twice.
                        infos[idx] = j
                    continue

                continue
            new_term_idxes.clear()

            to_collect, infos = self._choose_collectible(collectibles)
            if to_collect is None:
                break

            new_term_idx = self._collect(terms, infos)
            new_term_idxes.append(new_term_idx)

            continue
        # End Main loop.

        sum_node.evals = [
            _Sum(sum_node.exts, [i for i in terms if i is not None])
        ]
        return

    def _find_collectibles(self, exts, term):
        """Find the collectibles from a given term."""

        coeff, factor = self._parse_interm_ref(term)

        res = []  # type: typing.List[typing.Tuple[_Collectible, _CollectInfo]]

        if coeff != 1 and coeff != -1:
            # TODO: Add attempt to collect the coefficient.
            #
            # This could give some minor saving.
            pass

        prod_node = self._interms[factor.base]
        if len(prod_node.factors) > 2:
            # Single-factor does not offer collectible,
            # collectible * (something + 1) is so rare in real applications.

            if len(prod_node.evals) == 0:
                self._optimize_prod(prod_node)

            for eval_i in prod_node.evals:
                res.extend(self._find_collectibles_eval(
                    exts, eval_i
                ))
                continue

        return res

    def _find_collectibles_eval(self, exts, eval_: _Prod):
        """Get the collectibles for a particular evaluations of a product."""

        sums = eval_.sums
        factors = eval_.factors
        assert len(factors) == 2

        # Each evaluation could give two collectibles.
        res = []
        for lr in range(2):
            factor = factors[lr]
            collectible, ranges, coeff, substs = self._get_collectible_interm(
                exts, sums, factor
            )
            res.append((collectible, _CollectInfo(
                eval_=eval_, lr=lr,
                coeff=coeff, substs=substs, ranges=ranges
            )))
            continue

        return res

    def _get_collectible_interm(self, exts, sums, interm_ref):
        """Get a collectible from an intermediate reference."""

        terms = self._get_def(interm_ref)
        involved_symbs = interm_ref.atoms(Symbol)

        involved_exts = []
        other_exts = []
        for i, v in enumerate(exts):
            symb, range_ = v
            if symb in involved_symbs:
                involved_exts.append((
                    symb, range_.replace_label((_EXT, i, range_.label))
                ))
            else:
                other_exts.append((symb, range_))  # Undecorated.
            continue

        involved_sums = []
        for i, j in sums:
            # Sums not involved in both should be pushed in.
            assert (i in involved_symbs)
            involved_sums.append((
                i, j.replace_label((_SUMMED_EXT, j.label))
            ))
            continue

        coeff, key, all_sums = self._canon_terms(
            involved_exts + involved_sums, terms
        )
        ranges = _Ranges(
            involved_exts=self._write_in_orig_ranges(involved_exts),
            sums=self._write_in_orig_ranges(involved_sums),
            other_exts=other_exts
        )

        new_sums = (i for i in all_sums if i[1].label[0] == _SUMMED_EXT)
        return key, ranges, coeff, {
            i[0]: j[0]
            for i, j in zip(sums, new_sums)
            }

    def _choose_collectible(self, collectibles: _Collectibles):
        """Choose the most profitable collectible factor."""

        with_saving = (
            i for i in collectibles.items() if len(i[1]) > 1
        )

        try:
            return max(with_saving, key=lambda x: get_cost_key(
                self._get_collectible_saving(next(x[1].values()).ranges)
            ))
        except StopIteration:
            return None, None

    @staticmethod
    def _get_collectible_saving(ranges: _Ranges) -> Expr:
        """Get the saving factor for a collectible."""

        other_size = prod_(i.size for _, i in ranges.other_exts)
        sum_size = prod_(i.size for _, i in ranges.sums)
        ext_size = prod_(i.size for _, i in ranges.involved_exts)

        return other_size * add_costs(
            2 * sum_size * ext_size, ext_size, -sum_size
        )

    def _collect(self, terms, collect_infos: _CollectInfos):
        """Collect the given collectible factor.

        This function will mutate the given terms list.  Set one of the
        collected terms to the new sum term, whose index is going to be
        returned, with all the rest collected terms set to None.
        """

        residue_terms = []
        residue_exts = None
        new_term_idx = min(collect_infos.keys())
        for k, v in collect_infos.items():

            coeff, _ = self._parse_interm_ref(terms[k])
            eval_ = v.eval_
            coeff *= eval_.coeff * v.coeff  # Three levels of coefficients.

            residue_terms.extend(
                i.map(lambda x: coeff * x.xreplace(v.substs))
                for i in self._get_def(eval_.factors[0 if v.lr == 1 else 1])
            )

            curr_exts = list(sorted(itertools.chain(
                v.ranges.sums, v.ranges.other_exts
            )))
            if residue_exts is None:
                residue_exts = curr_exts
            else:
                assert residue_exts == curr_exts

            continue

        new_ref = self._form_sum_interm(residue_exts, residue_terms)

        for k, v in collect_infos.items():
            if k == new_term_idx:
                terms[k] = self._form_collected(terms[k], v, new_ref)
            else:
                terms[k] = None

        return new_term_idx

    def _form_collected(self, term, info: _CollectInfo, new_ref) -> Expr:
        """Form new sum term with some factors collected based on a term.
        """

        eval_ = info.eval_
        collected_factor = eval_.factors[info.lr].xreplace(info.substs)

        interm_coeff, interm = self._parse_interm_ref(new_ref)
        coeff = interm_coeff / info.coeff

        _, orig_node = self._parse_interm_ref(term)
        orig_exts = orig_node.exts

        new_node = _Prod(orig_exts, orig_node.sums, coeff, [
            collected_factor, interm
        ])
        new_node.evals = [new_node]

        base = self._get_next_internal(len(orig_exts) == 0)
        self._interms[base] = new_node

        return base[tuple(i for i, _ in orig_exts)]

    #
    # Product optimization.
    #

    def _optimize_prod(self, prod_node):
        """Optimize the product evaluation node."""


#
# Utility constants.
#


_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)

_EXT = 0
_SUMMED_EXT = 1
_SUMMED = 2

#
# Small type definitions.
#


_Grain = collections.namedtuple('_Grain', [
    'exts',
    'terms'
])

#
# The information on collecting a collectible.
#
# Interpretation, after the substitutions given in ``substs``, the ``lr`` factor
# in the evaluation ``eval_`` will be turned into ``coeff`` times the actual
# collectible.
#

_CollectInfo = collections.namedtuple('_Residue', [
    'eval_',
    'lr',
    'coeff',
    'substs',
    'ranges'
])

_Ranges = collections.namedtuple('_Ranges', [
    'involved_exts',
    'sums',
    'other_exts'
])

_Collectible = typing.Tuple[Term, ...]

_CollectInfos = typing.Dict[int, _CollectInfo]

_Collectibles = typing.Dict[_Collectible, _CollectInfos]


#
# Core evaluation DAG nodes.
#


class _EvalNode:
    """A node in the evaluation graph.
    """

    def __init__(self, exts):
        """Initialize the evaluation node.
        """

        self.exts = exts
        self.evals = []  # type: typing.List[_EvalNode]
        self.n_refs = 0


class _Sum(_EvalNode):
    """Sum nodes in the evaluation graph."""

    def __init__(self, exts, sum_terms):
        """Initialize the node."""
        super().__init__(exts)
        self.sum_terms = sum_terms


class _Prod(_EvalNode):
    """Product nodes in the evaluation graph.
    """

    def __init__(self, exts, sums, coeff, factors):
        """Initialize the node."""
        super().__init__(exts)
        self.sums = sums
        self.coeff = coeff
        self.factors = factors
        self.canon = None
