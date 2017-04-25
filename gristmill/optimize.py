"""Optimizer for the contraction computations."""

import collections
import heapq
import itertools
import typing
import warnings

from drudge import TensorDef, prod_, Term, Range, sum_
from sympy import (
    Integer, Symbol, Expr, IndexedBase, Mul, Indexed, sympify, primitive, Wild
)
from sympy.utilities.iterables import multiset_partitions

from .utils import get_cost_key, add_costs, get_total_size, DSF


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
# Internal small type definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


#
# Small type definitions.
#
# These named tuples should be upgraded when PySpark has support for Python 3.6
# in their stable version.


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
# The information on collecting a collectible.
#
# Interpretation, after the substitutions given in ``substs``, the ``lr`` factor
# in the evaluation ``eval_`` will be turned into ``coeff`` times the actual
# collectible.
#

_CollectInfo = collections.namedtuple('_CollectInfo', [
    'eval_',
    'lr',
    'coeff',
    'substs',
    'ranges',
    'add_cost'
])

_Ranges = collections.namedtuple('_Ranges', [
    'involved_exts',
    'sums',
    'other_exts'
])

_Collectible = typing.Tuple[Term, ...]

_CollectInfos = typing.Dict[int, _CollectInfo]

_Collectibles = typing.Dict[typing.Tuple[_Collectible, _Ranges], _CollectInfos]

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
                terms = self._index_prod(eval_, ref.indices)
                assert len(terms) == 1
                term = terms[0]
                factors, term_coeff = term.get_amp_factors(self._interms)

                # Switch back to evaluation node for using the facilities for
                # product nodes.
                new_term, term_deps = self._form_prod_def_term(_Prod(
                    term_node.base, exts, term.sums,
                    ref.coeff * term_coeff, factors
                ))

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
            base = comput.base
            exts = tuple((s, self._input_ranges[r]) for s, r in comput.exts)
            terms = [
                i.map(proc_amp, sums=tuple(
                    (s, self._input_ranges[r]) for s, r in i.sums
                )) for i in comput.terms
            ]

            interm_base = base if len(exts) == 0 else base.args[0]
            if interm_base in self._interms:

                if len(terms) == 1 and len(terms[0].sums) == 0:
                    # Remove shallow intermediates.  The saving might be too
                    # modest to justify the additional memory consumption.
                    #
                    # TODO: Move it earlier to a better place.
                    repl_lhs = base[tuple(
                        _WILD_FACTORY[i] for i, _ in enumerate(exts)
                    )] if len(exts) > 0 else base
                    repl_rhs = proc_amp(terms[0].amp.xreplace(
                        {v[0]: _WILD_FACTORY[i] for i, v in enumerate(exts)}
                    ))
                    repls.append((repl_lhs, repl_rhs))
                    continue  # No new intermediate added.

                final_base = type(base)(self._interm_fmt.format(next_idx))
                next_idx += 1
                substs[base] = final_base
            else:
                final_base = base

            if len(exts) > 0:
                final_base = IndexedBase(final_base)

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

        # Now we embark upon the heroic factorization.
        collectibles = collections.defaultdict(dict)  # type: _Collectibles
        while True:

            for idx in new_term_idxes:
                term = terms[idx]

                # Loop over collectibles the new term can offer.
                for i, j in self._find_collectibles(exts, term):
                    infos = collectibles[i, j.ranges]
                    if idx not in infos:
                        # The same term cannot provide the same collectible
                        # twice.
                        infos[idx] = j
                    continue

                continue
            new_term_idxes.clear()

            to_collect, infos, total_cost = self._choose_collectible(
                collectibles
            )
            if to_collect is None:
                break

            new_term_idx = self._collect(terms, infos, total_cost)
            new_term_idxes.append(new_term_idx)

            del collectibles[to_collect]
            for i in infos.keys():
                for j in collectibles.values():
                    if i in j:
                        del j[i]

            continue
        # End Main loop.

        rem_terms = [i for i in terms if i is not None]
        sum_node.evals = [_Sum(
            sum_node.base, sum_node.exts, rem_terms
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
            coeff, ref = self._parse_interm_ref(term)
            if ref is None:
                plain_scalars.append(coeff)
                continue
            elif isinstance(ref, Symbol):
                base = ref
                indices = ()
            elif isinstance(ref, Indexed):
                base = ref.base
                indices = ref.indices
            else:
                assert False

            interm_refs[base][indices] += coeff
            continue

        # Intermediate referenced only once goes to the result directly and wait
        # to be factored, others wait to be pulled and do not participate in
        # factorization.
        res_terms = plain_scalars
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
                    (k[indices] if len(indices) > 0 else k) * coeff
                )
            else:
                # Here we use name for sorting directly, since here we cannot
                # have general expressions hence no need to use the expensive
                # sort_key.
                raw = list(sorted(v.items(), key=lambda x: tuple(
                    i.name for i in x[0]
                )))
                leading_coeff = raw[0][1]
                pull_info[tuple(
                    (i, j / leading_coeff) for i, j in raw
                )].append((k, leading_coeff))

        # Now we treat the terms from which new intermediates might be pulled
        # out.
        for k, v in pull_info.items():
            pivot = k[0][0]
            assert k[0][1] == 1
            if len(v) == 1:
                # No need to form a new intermediate.
                base, coeff = v[0]
                pivot_ref = base[pivot] * coeff
            else:
                # We need to form an intermediate here.
                interm_exts = tuple(
                    (i, exts_dict[i]) for i in pivot
                )
                pivot_ref, interm_node = self._form_sum_interm(interm_exts, [
                    term.scale(coeff)
                    for base, coeff in v
                    for term in self._get_def(base[pivot])
                ])
                self._optimize(interm_node)

            for indices, coeff in k:
                substs = {
                    i: j for i, j in zip(pivot, indices)
                }
                res_terms.append(
                    pivot_ref.xreplace(substs) * coeff / k[0][1]
                )
                continue

            continue

        return res_terms, res_collectible_idxes

    def _find_collectibles(self, exts, term):
        """Find the collectibles from a given term.

        Collectibles are going to be yielded as key and infos pairs.
        """

        res = []  # type: typing.List[typing.Tuple[_Collectible, _CollectInfo]]

        coeff, ref = self._parse_interm_ref(term)
        if ref is None:
            return res

        if coeff != 1 and coeff != -1:
            # TODO: Add attempt to collect the coefficient.
            #
            # This could give some minor saving.
            pass

        prod_node = self._interms[
            ref.base if isinstance(ref, Indexed) else ref
        ]
        self._optimize(prod_node)

        if len(prod_node.factors) > 1:
            # Single-factor does not offer collectible,
            # collectible * (something + 1) is so rare in real applications.

            for eval_i in prod_node.evals:
                res.extend(self._find_collectibles_eval(
                    exts, ref, eval_i, prod_node.total_cost
                ))
                continue

        return res

    def _find_collectibles_eval(
            self, exts, ref: Expr, eval_: _Prod, opt_cost
    ):
        """Get the collectibles for a particular evaluations of a product."""

        # To begin, we first need to substitute the external indices in for this
        # particular evaluation inside its ambient.
        total_cost = eval_.total_cost
        assert total_cost is not None
        if len(eval_.exts) == 0:
            assert isinstance(ref, Symbol)
        else:
            assert isinstance(ref, Indexed)
            eval_terms = self._index_prod(eval_, ref.indices)
            assert len(eval_terms) == 1
            eval_term = eval_terms[0]
            factors, coeff = eval_term.get_amp_factors(self._interms)
            eval_ = _Prod(
                _SUBSTED_EVAL_BASE, exts, eval_term.sums, coeff, factors
            )
            eval_.total_cost = total_cost

        sums = eval_.sums
        factors = eval_.factors
        assert len(factors) == 2

        # Each evaluation could give two collectibles.
        res = []
        for lr in range(2):
            factor = factors[lr]
            other_factor = factors[0 if lr == 1 else 1]
            collectible, ranges, coeff, substs = self._get_collectible_interm(
                exts, sums, factor, other_factor
            )
            res.append((collectible, _CollectInfo(
                eval_=eval_, lr=lr,
                coeff=coeff, substs=substs, ranges=ranges,
                add_cost=eval_.total_cost - opt_cost
            )))
            continue

        return res

    def _get_collectible_interm(self, exts, sums, interm_ref, other_ref):
        """Get a collectible from an intermediate reference."""

        terms = self._get_def(interm_ref)
        involved_symbs = interm_ref.atoms(Symbol)
        other_symbs = other_ref.atoms(Symbol)

        involved_exts = []
        other_exts = []
        for i, v in enumerate(exts):
            symb, range_ = v
            if symb in involved_symbs:
                involved_exts.append((
                    symb, range_.replace_label((range_.label, _EXT, i))
                ))
            if symb in other_symbs:
                other_exts.append((symb, range_))  # Undecorated.
            continue

        involved_sums = []
        for i, j in sums:
            # Sums not involved in both should be pushed in.
            assert i in involved_symbs
            involved_sums.append((
                i, j.replace_label((j.label, _SUMMED_EXT))
            ))
            continue

        coeff, key, all_sums = self._canon_terms(
            tuple(itertools.chain(involved_exts, involved_sums)), terms
        )
        ranges = _Ranges(
            involved_exts=self._write_in_orig_ranges(involved_exts),
            sums=self._write_in_orig_ranges(involved_sums),
            other_exts=self._write_in_orig_ranges(other_exts)
        )

        new_sums = (i for i in all_sums if i[1].label[1] == _SUMMED_EXT)
        return key, ranges, coeff, {
            i[0]: j[0]
            for i, j in zip(involved_sums, new_sums)
        }

    def _choose_collectible(self, collectibles: _Collectibles):
        """Choose the most profitable collectible factor.

        The collectible, its infos, and the final cost of the evaluation after
        the collection will be returned.
        """

        with_saving = (
            i for i in collectibles.items() if len(i[1]) > 1
        )

        optimal = None
        new_total_cost = None
        largest_saving = None
        for key, infos in with_saving:
            collectible, ranges = key
            raw_saving = self._get_collectible_saving(ranges)
            saving = raw_saving - sum(
                i.add_cost for i in infos.values()
            )
            saving_key = get_cost_key(saving)

            if_save = len(saving_key[1]) > 0 and saving_key[1][0] > 0
            if_better = (
                largest_saving is None or saving_key > largest_saving[1]
            )

            if if_save and if_better:
                largest_saving = (saving, saving_key)
                optimal = ((collectible, ranges), infos)
                orig_cost = sum(i.eval_.total_cost for i in infos.values())
                new_total_cost = orig_cost - raw_saving

            continue

        if optimal is None:
            return None, None, None
        else:
            return optimal[0], optimal[1], new_total_cost

    @staticmethod
    def _get_collectible_saving(ranges: _Ranges) -> Expr:
        """Get the saving factor for a collectible."""

        other_size = get_total_size(ranges.other_exts)
        sum_size = get_total_size(ranges.sums)
        ext_size = get_total_size(ranges.involved_exts)

        return other_size * add_costs(
            2 * sum_size * ext_size, ext_size, -sum_size
        )

    def _collect(self, terms, collect_infos: _CollectInfos, new_cost):
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
                i.map(lambda x: coeff * x) for i in self._get_def(
                    eval_.factors[0 if v.lr == 1 else 1].xreplace(v.substs)
                )
            )

            curr_exts = tuple(
                itertools.chain(v.ranges.other_exts, v.ranges.sums),
            )
            if residue_exts is None:
                residue_exts = curr_exts
            else:
                assert residue_exts == curr_exts

            continue

        new_ref, _ = self._form_sum_interm(residue_exts, residue_terms)

        for k, v in collect_infos.items():
            if k == new_term_idx:
                terms[k] = self._form_collected(terms[k], v, new_ref, new_cost)
            else:
                terms[k] = None

        return new_term_idx

    def _form_collected(
            self, term, info: _CollectInfo, new_ref, new_cost
    ) -> Expr:
        """Form new sum term with some factors collected based on a term.
        """

        eval_ = info.eval_
        collected_factor = eval_.factors[info.lr].xreplace(info.substs)

        interm_coeff, interm = self._parse_interm_ref(new_ref)
        coeff = interm_coeff / info.coeff

        _, orig_ref = self._parse_interm_ref(term)
        orig_node = self._interms[
            orig_ref.base if isinstance(orig_ref, Indexed) else orig_ref
        ]
        orig_exts = orig_node.exts

        base = self._get_next_internal(len(orig_exts) == 0)

        new_node = _Prod(base, orig_exts, eval_.sums, coeff, [
            collected_factor, interm
        ])
        new_node.total_cost = new_cost
        new_node.evals = [new_node]

        self._interms[base] = new_node

        return (
            base[tuple(i for i, _ in orig_exts)]
            if isinstance(base, IndexedBase) else base
        )

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
                    if self._strategy == Strategy.BEST:
                        evals.clear()

                # New optimal is always added.
                def need_add_eval():
                    """If the current evaluation need to be added."""
                    if self._strategy == Strategy.BEST:
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
            final_cost = self._get_prod_final_cost(
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

    @staticmethod
    def _get_prod_final_cost(exts_total_size, sums_total_size) -> Expr:
        """Compute the final cost for a pairwise product evaluation."""

        if sums_total_size == 1:
            return exts_total_size
        else:
            return _TWO * exts_total_size * sums_total_size

    def _form_prod_eval(
            self, prod_node: _Prod, broken_sums, parts: typing.Tuple[_Part, ...]
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
