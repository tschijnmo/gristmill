"""Optimizer for the contraction computations."""

import collections
import heapq
import itertools
import typing
import warnings

from drudge import TensorDef, prod_, Term, Range
from sympy import Integer, Symbol, Expr, IndexedBase, Mul, Indexed
from sympy.utilities.iterables import multiset_partitions

from .utils import get_cost_key, add_costs, DSF


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
        self._range_var = None  # The only variable for range sizes.
        self._excl = set()
        self._input_ranges = {}  # Substituted range to original range.

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
                raise ValueError(
                    'Invalid term to optimize', term, 'expecting scalar'
                )
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
        of the object will also be updated.  User error will also be reported.
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
        """Write the summations in terms of undecorated bare ranges.

        The labels in the ranges are assumed to be decorated.
        """
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

        canon_new_sums = canon_sums[:n_new]
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
        instance in a term in an summation node, is very rigid.
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
        else:
            return self._form_sum_from_terms(exts, terms)

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
            self._interms[base] = interm

        return coeff * base[tuple(
            i for i, _ in canon_exts
        )]

    def _form_sum_interm(self, exts, terms) -> Expr:
        """Form a sum intermediate.
        """

        decored_exts = tuple(
            (i, j.replace_label((_EXT, j.label)))
            for i, j in exts
        )
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
        """Optimize the product evaluation node.
        """

        assert len(prod_node.evals) == 0
        n_factors = len(prod_node.factors)

        if n_factors < 2:
            assert n_factors == 1
            prod_node.evals.append(prod_node)
            prod_node.total_cost = self._get_prod_final_cost(
                prod_(i.size for _, i in prod_node.exts),
                prod_node.sums
            )
            return

        evals = prod_node.evals
        curr_total = None
        for final_cost, parts_gen in self._gen_factor_parts(prod_node):
            if_break = (
                curr_total is not None
                and get_cost_key(final_cost) > curr_total[0]
            )
            if if_break:
                break
            # Else

            for parts in parts_gen:

                # Recurse.
                for i in parts:
                    self._optimize_prod(i.node)
                    continue

                total_cost = (
                    final_cost
                    + parts[0].node.total_cost
                    + parts[1].node.total_cost
                )
                total_cost_key = get_cost_key(total_cost)
                if curr_total is None or curr_total[0] > total_cost_key:
                    curr_total = (total_cost_key, total_cost)
                    evals.clear()
                    evals.append(self._form_prod_eval(prod_node, parts))
                elif curr_total[0] == total_cost_key:
                    evals.append(self._form_prod_eval(prod_node, parts))

                continue

        prod_node.total_cost = curr_total[1]
        return

    def _gen_factor_parts(self, prod_node: _Prod):
        """Generate all the partitions of factors in a product node."""

        # Compute things invariant to different summations for performance.
        exts = prod_node.exts
        exts_total_size = prod_(i.size for _, i in exts)

        factor_atoms = [i.atoms(Symbol) for i in prod_node.factors]

        sum_involve = [
            {j for j, v in enumerate(factor_atoms) if i in v}
            for i, _ in prod_node.sums
            ]

        dumm2index = tuple(
            {v[0]: j for j, v in enumerate(i)}
            for i in [prod_node.exts, prod_node.sums]
        )
        factor_infos = [
            tuple([i[j] for j in atoms if j in i] for i in dumm2index)
            for atoms in factor_atoms
            ]

        for kept in self._gen_broken_kept_sums(prod_node.sums):
            final_cost = self._get_prod_final_cost(
                exts_total_size,
                [i for i, j in zip(prod_node.sums, kept) if not j]
            )
            yield final_cost, self._gen_parts_w_kept_sums(
                prod_node, kept, sum_involve, factor_infos
            )
            continue

    @staticmethod
    def _gen_broken_kept_sums(sums):
        """Generate kept summations in increasing size of broken summations.

        The results will be given as boolean array giving if the corresponding
        entry is to be kept.
        """

        sizes = [i.size for _, i in sums]
        n_sums = len(sizes)

        def get_size(kept):
            """Get the total broken summation size."""
            return prod_(i for i, j in zip(sizes, kept) if not j)

        init = list(range(n_sums))  # Everything is kept.
        queue = [(get_size(init), init)]
        while len(queue) > 0:
            curr = heapq.heappop(queue)
            yield curr
            curr_kept = curr[1]
            for i in range(n_sums):
                if curr_kept[i]:
                    new_kept = list(curr_kept)
                    new_kept[i] = False
                    heapq.heappush(
                        queue,
                        (get_size(new_kept), new_kept)
                    )
                    continue
                else:
                    break
            continue

    def _gen_parts_w_kept_sums(
            self, prod_node: _Prod, kept, sum_involve, factor_infos
    ):
        """Generate all partitions with given summations kept."""

        dsf = DSF(i for i, _ in enumerate(factor_infos))

        for i, j in zip(kept, sum_involve):
            if i:
                dsf.union(j)
            continue

        chunks = dsf.sets
        if len(chunks) < 2:
            return

        for part in self._gen_parts_from_chunks(kept, chunks, sum_involve):
            yield tuple(
                self._form_part(prod_node, i, sum_involve, factor_infos)
                for i in part
            )

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

                # Now we have broken sum.
                involve = sum_involve[i]
                if any(part.isdisjoint(involve) for part in factors_part):
                    break
            else:
                yield factors_part

    def _form_part(self, prod_node, factor_idxes, sum_involve, factor_infos):
        """Form a partition for the given factors."""

        involved_exts, involved_sums = [
            set.union(*[i[label] for i in factor_infos])
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
            if sum_involve[i].issubset(involved_sums):
                sums.append(v)
            else:
                exts.append(v)
            continue

        ref, node = self._form_prod_interm(exts, sums, factors)
        return _Part(ref=ref, node=node)

    @staticmethod
    def _get_prod_final_cost(exts_total_size, sums) -> Expr:
        """Compute the final cost for a pairwise product evaluation."""

        if len(sums) == 0:
            return exts_total_size
        else:
            return _TWO * exts_total_size * prod_(i.size for _, i in sums)

    def _form_prod_eval(
            self, prod_node: _Prod, parts: typing.Tuple[_Part, ...]
    ):
        """Form an evaluation for a product node."""

        assert len(parts) == 2

        coeff = _UNITY
        factors = []
        for i in parts:
            curr_coeff, curr_ref = self._parse_interm_ref(i.ref)
            coeff *= curr_coeff
            factors.append(curr_ref)
            continue

        return _Prod(prod_node.exts, prod_node.sums, coeff, factors)


#
# Utility constants.
#


_UNITY = Integer(1)
_NEG_UNITY = Integer(-1)
_TWO = Integer(2)

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

_Part = collections.namedtuple('_Part', [
    'ref',
    'node'
])


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
        self.total_cost = None
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
