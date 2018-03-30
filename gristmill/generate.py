"""Generate source code from optimized computations."""

import abc
import collections
import functools
import itertools
import textwrap
import types
import typing

from drudge.term import try_resolve_range
from sympy import (
    Expr, Mul, Pow, Integer, Rational, Add, Indexed, IndexedBase
)
from sympy.printing.ccode import CCodePrinter
from sympy.printing.fcode import FCodePrinter
from sympy.printing.printer import Printer
from sympy.printing.python import PythonPrinter

from drudge import TensorDef, Term, Range, prod_
from .utils import JinjaEnv


#
# General description of events
# -----------------------------
#

class TensorComp:
    """Full description of a tensor computation.

    Although this does not actually form an event by itself, it is referenced in
    all the events for the actual computation.

    """

    def __init__(self, is_interm: bool, def_: TensorDef, ctx):
        """Initialize the computation."""
        self._is_interm = is_interm
        self._def = def_
        self._ctx = ctx

    @property
    def is_interm(self):
        """If the computation is an intermediate.
        """
        return self._is_interm

    @property
    def def_(self):
        """The tensor definition for the computation."""
        return self._def

    @property
    def ctx(self):
        """The rendering context for the entire computation.
        """
        return self._ctx

    @property
    def target(self):
        """The base of the target tensor to compute.
        """
        return self._def.base

    def __str__(self):
        """Form a string representation.

        For the ease of readability during debugging, here the string form is
        only based on the target tensor name.
        """
        return str(self.target)


class TensorDecl(typing.NamedTuple):
    """Events for declaration of intermediate tensors.
    """

    comput: TensorComp

    def __repr__(self):
        """Form a string, mostly for debugging.
        """
        return '_TensorDecl({!s})'.format(self.comput)


class BeforeComp(typing.NamedTuple):
    """Events that come before the first computation of any tensor.

    Normally, it should be rendered as memory allocation or initialization to
    zero.
    """

    comput: TensorComp

    def __repr__(self):
        """Form a string, mostly for debugging."""
        return '_BeforeCompute({!s})'.format(self.comput)


class CompTerm(typing.NamedTuple):
    """Events for the computation of a term in a tensor.

    This events attempt to add a term to the LHS of the computation.

    Attributes
    ----------

    comput
       The actual full computation.

    term_idx
        Index to the term being computed.

    term_ctx
        The rendering context for the current term.

    """

    comput: TensorComp
    term_idx: int
    term_ctx: types.SimpleNamespace

    def __repr__(self):
        """Format a string, mostly for debugging."""
        return "_ComputeTerm({!s}, {!s})".format(
            self.comput, self.term_ctx.orig_term
        )


class OutOfUse(typing.NamedTuple):
    """Events after intermediate tensors are no longer in use.

    This event can be used for the freeing of the associated computer memory.
    """

    comput: TensorComp

    def __repr__(self):
        """Form a string, mostly for debugging."""
        return '_NoLongerInUse({!s})'.format(self.comput)


#
# Actual printers
# ---------------
#


class BasePrinter(abc.ABC):
    """The base class for tensor printers.
    """

    def __init__(
            self, scal_printer: Printer, indexed_proc_cb=lambda x: None,
            **kwargs
    ):
        """Initializes a base printer.

        Parameters
        ----------

        scal_printer
            The SymPy printer for scalar quantities.

        indexed_proc_cb
            It is going to be called with context nodes with ``base`` and
            ``indices`` (in both the root and for each indexed factors, as
            described in :py:meth:`transl`) to do additional processing.  For
            most tasks, :py:func:`mangle_base` can be helpful.

        kwargs
            All the keyword arguments are forwarded to :py:cls:`utils.JinjaEnv`
            constructor.

        """

        env = JinjaEnv(**kwargs)

        self._env = env
        self._scal_printer = scal_printer
        self._indexed_proc = indexed_proc_cb

    #
    # Translation to rendering contexts
    # ---------------------------------
    #

    def transl(self, tensor_def: TensorDef) -> types.SimpleNamespace:
        """Translate tensor definition into context for template rendering.

        This function will translate the given tensor definition into a simple
        namespace that could be easily used as the context in the actual Jinja
        template rendering.

        The context contains fields,

        base
            A printed form for the base of the tensor definition.

        indices
            A list of external indices.  For each entry, keys ``index`` and
            ``range`` are present to give the printed form of the index and the
            range object that it is over. For convenience, ``lower``, ``upper``,
            and ``size`` have the printed form of lower/upper bounds and the
            size of the range.  We also have ``lower_expr``, ``upper_expr``, and
            ``size_expr`` for the unprinted expression of them.

        n_exts
            The number of external indices for the LHS.

        terms
            A list of terms for the tensor, with each entry being a simple
            namespace with keys,

            sums
                A list of summations in the tensor term.  Its entries are in the
                same format as the external indices for tarrays.

            phase
                ``+`` sign or ``-`` sign.  For the phase of the term.

            numerator
                The printed form of the numerator of the coefficient of the
                term.  It can be a simple ``1`` string.

            denominator
                The printed form of the denominator.

            indexed_factors
                The indexed factors of the term.  Each is given as a simple
                namespace with key ``base`` for the printed form of the base,
                and a key ``indices`` giving the indices to the key, in the same
                format as the ``indices`` field of the base context.

            other_factors
                Factors which are not simple indexed quantity, given as a list
                of the printed form directly.

            orig_term
                The original term object.

        orig_def
            The original definition of the tensor computation.

        The actual content of the context can also be customized by overriding
        the :py:meth:`proc_ctx` in subclasses.

        """

        ctx = types.SimpleNamespace()
        ctx.orig_def = tensor_def

        base = tensor_def.base
        ctx.base = self._print_scal(
            base.label if isinstance(base, IndexedBase) else base
        )
        ctx.indices = self._form_indices_ctx(tensor_def.exts)
        ctx.n_exts = len(ctx.indices)

        # The stack keeping track of the external and internal indices for range
        # resolution.
        indices_dict = dict(tensor_def.exts)
        resolvers = tensor_def.rhs.drudge.resolvers.value

        terms = []
        ctx.terms = terms

        # Render each term in turn.
        for term in tensor_def.rhs_terms:

            term_ctx = types.SimpleNamespace()
            term_ctx.orig_term = term
            terms.append(term_ctx)

            indices_dict.update(term.sums)
            term_ctx.sums = self._form_indices_ctx(term.sums)

            factors, coeff = term.get_amp_factors(monom_only=False)

            coeff = coeff.together()
            if isinstance(coeff, Mul):
                coeff_factors = coeff.args
            else:
                coeff_factors = (coeff,)

            phase = 1
            numerator = []
            denominator = []
            for factor in coeff_factors:
                if isinstance(factor, Integer):
                    if factor.is_negative:
                        phase *= -1
                        factor = -factor
                    if factor != 1:
                        numerator.append(factor)
                elif isinstance(factor, Rational):
                    for i, j in [
                        (factor.p, numerator), (factor.q, denominator)
                    ]:
                        if i < 0:
                            phase *= -1
                            i = -i
                        if i != 1:
                            j.append(i)
                elif isinstance(factor, Pow) and factor.args[1].is_negative:
                    denominator.append(1 / factor)
                else:
                    numerator.append(factor)
                continue

            term_ctx.phase = '+' if phase == 1 else '-'
            for i, j, k in [
                (numerator, 'numerator', Add),
                (denominator, 'denominator', (Add, Mul))
            ]:
                val = prod_(i)
                printed_val = self._print_scal(val)
                if isinstance(val, k):
                    printed_val = '(' + printed_val + ')'
                setattr(term_ctx, j, printed_val)
                continue

            indexed_factors = []
            term_ctx.indexed_factors = indexed_factors
            other_factors = []
            term_ctx.other_factors = other_factors
            other_factors_expr = []
            term_ctx.other_factors_expr = other_factors_expr
            for factor in factors:

                if isinstance(factor, Indexed):
                    factor_ctx = types.SimpleNamespace()
                    factor_ctx.base = self._print_scal(factor.base.label)
                    factor_ctx.indices = self._form_indices_ctx((
                        (i, try_resolve_range(i, indices_dict, resolvers))
                        for i in factor.indices
                    ), enforce=False)
                    indexed_factors.append(factor_ctx)
                else:
                    other_factors_expr.append(factor)
                    other_factors.append(self._print_scal(factor))

            self.proc_ctx(tensor_def, term, ctx, term_ctx)

            for i, _ in term.sums:
                del indices_dict[i]
            continue

        self.proc_ctx(tensor_def, None, ctx, None)

        return ctx

    def proc_ctx(
            self, tensor_def: TensorDef, term: typing.Optional[Term],
            tensor_entry: types.SimpleNamespace,
            term_entry: typing.Optional[types.SimpleNamespace]
    ):
        """Make additional processing of the rendering context.

        This method can be override to make additional processing on the
        rendering context described in :py:meth:`transl` to perform additional
        customization or to make more information available.

        It will be called for each of the terms during the processing.  And
        finally it will be called again with the term given as None for a final
        processing.

        By default, the indexed quantities nodes are processed by the user-given
        call-back.
        """

        if term is None:
            self._indexed_proc(tensor_entry)
        else:
            for i in term_entry.indexed_factors:
                self._indexed_proc(i)
                continue
        return

    def render(self, templ_name: str, ctx: types.SimpleNamespace) -> str:
        """Render the given context for the given template.

        Meaningful subclass methods can call this function for actual
        functionality.
        """

        templ = self._env.get_template(templ_name)
        return templ.render(ctx.__dict__)

    def _form_indices_ctx(
            self,
            pairs: typing.Iterable[typing.Tuple[Expr, typing.Optional[Range]]],
            enforce=True
    ):
        """Form indices context.
        """

        res = []
        for index, range_ in pairs:

            if range_ is None or not range_.bounded:
                if enforce:
                    raise ValueError(
                        'Invalid range to print', range_, 'for', index,
                        'expecting a bounded range.'
                    )
                else:
                    lower = None
                    upper = None
                    size = None
                    lower_expr = None
                    upper_expr = None
                    size_expr = None
            else:
                lower_expr = range_.lower
                upper_expr = range_.upper
                size_expr = range_.size
                lower = self._print_scal(lower_expr)
                upper = self._print_scal(upper_expr)
                size = self._print_scal(size_expr)

            res.append(types.SimpleNamespace(
                index=self._print_scal(index), range=range_,
                lower=lower, upper=upper, size=size,
                lower_expr=lower_expr, upper_expr=upper_expr,
                size_expr=size_expr
            ))
            continue

        return res

    def _print_scal(self, expr: Expr):
        """Print a scalar."""
        return self._scal_printer.doprint(expr)

    #
    # Formation of event lists from computations
    # ------------------------------------------
    #

    def form_events(self, defs: typing.Iterable[TensorDef], origs=None):
        """Form a linear list of full events from the definitions.

        This is a mostly developer method that can turn any list of tensor
        computations into a full list of events for their computation.

        Currently, the events are comprised of

        - Declarations of all intermediates,

        - Events before the first computation of a tensor,

        - Addition of a term, which maybe a plain reference to another tensor or
          a contraction, to another tensor,

        - Events indicating that an intermediate is no longer used.

        Notably, we do not have declaration events for non-intermediate tensors,
        since they generally have different treatment from the intermediates.

        Parameters
        ----------

        defs:
            The computations.

        origs:
            An optional iterable of the original tensor computations before the
            optimization.  Computations of bases out of this iterable are
            understood to be intermediates.  When it is not given, no
            computation will be taken as intermediate.


        Notes
        -----

        Internally, some attributes are attached to computations, which will be
        deleted after the procedure.  These attributes generally use integer
        indices to track computations and terms.  For either a computation or a
        term, it is called pending when it is not yet computed, otherwise it is
        said to be computed or finished.

        pends
            The indices to the pending terms of the computation.  When it is
            empty, the computation is finished.

        deps
            Pending terms that is dependent on the current tensor.  For
            intermediates only. Given as computation index / term index pairs.
            Earlier dependents comes first.  Due to the absence of native
            support of an ordered set or linked list, here it is an ordered
            dictionary with all the values being None.

        For each term in the context, some attributes are also added.

        pend_prereqs
            Pending prerequisites for the evaluation of a terms, given as a set
            of computation indices.

        fin_prereqs
            Finished prerequisites for a term.

        All these attributes are for the generation of event lists only.  They
        will be removed after the procedure.  So printers cannot read them and
        do not need to.

        """

        if origs is None:
            orig_bases = None
        else:
            orig_bases = {i.base for i in origs}

        # Generation of _TensorComput objects and rendering contexts.
        computs = []
        base2idx = {}
        interms = set()  # Bases for the intermediates
        for idx, def_ in enumerate(defs):
            base = def_.base
            is_interm = origs is not None and base not in orig_bases
            comput = TensorComp(
                is_interm=is_interm, def_=def_, ctx=self.transl(def_)
            )

            # Internal attributes.
            #
            # To be actually filled later, except pends.
            comput.pends = {i for i in range(def_.n_terms)}
            comput.deps = collections.OrderedDict()
            for i in comput.ctx.terms:
                i.pend_prereqs = set()
                i.fin_prereqs = set()

            computs.append(comput)
            base2idx[base] = idx
            if is_interm:
                interms.add(base)
            continue

        # Track the dependencies for the intermediates.
        for comput_idx, comput in enumerate(computs):
            for term_idx, term_ctx in enumerate(comput.ctx.terms):
                term: Term = term_ctx.orig_term
                for b in interms:
                    if term.has_base(b):
                        b_idx = base2idx[b]
                        term_ctx.pend_prereqs.add(b_idx)
                        computs[b_idx].deps[
                            (comput_idx, term_idx)
                        ] = None
                    continue
                continue
            continue

        # Actual generation of the event list.
        events = []
        for i in computs:
            if i.is_interm:
                events.append(TensorDecl(comput=i))
            continue

        for idx, comput in enumerate(computs):
            while len(comput.pends) > 0:
                term_idx = next(iter(comput.pends))
                self._add_term_eval(events, computs, idx, term_idx)
                continue
            continue

        # Remove temporary attributes.
        for i in computs:
            assert len(i.pends) == 0
            del i.pends
            assert len(i.deps) == 0
            del i.deps
            for j in i.ctx.terms:
                assert len(j.pend_prereqs) == 0
                del j.pend_prereqs
                del j.fin_prereqs
                continue
            continue

        return events

    def _add_term_eval(
            self, events, computs: typing.Sequence[TensorComp], comput_idx,
            term_idx
    ):
        """Add the evaluation of a term to the events list.

        This function also attempts to add the preparation steps and following
        steps.  Also the given term index is going to be removed from the
        pending terms of the computation as well.
        """

        comput = computs[comput_idx]
        pends = comput.pends
        term_ctxes = comput.ctx.terms
        if len(pends) == len(term_ctxes):
            # First time a term is to be computed.
            events.append(BeforeComp(comput))

        term_ctx = term_ctxes[term_idx]
        if len(term_ctx.pend_prereqs) != 0:
            raise ValueError(
                'Invalid evaluation sequence!', comput.target, 'needs',
                [computs[i].target for i in term_ctx.pend_prereqs]
            )

        events.append(CompTerm(
            comput=comput, term_idx=term_idx, term_ctx=term_ctx
        ))

        # Possibly free dependent intermediates.
        for i in term_ctx.fin_prereqs:
            prereq = computs[i]
            del prereq.deps[(comput_idx, term_idx)]  # Must be present.
            if prereq.is_interm and len(prereq.deps) == 0:
                events.append(OutOfUse(prereq))
            continue

        # When this is the last term.  Possibly drive the evaluation of some
        # intermediates that can be evaluated.
        pends.remove(term_idx)
        if len(pends) == 0:
            deps = list(comput.deps.keys())
            for dc_i, dt_i in deps:
                dc = computs[dc_i]
                dt = dc.ctx.terms[dt_i]

                pend_prereqs = dt.pend_prereqs
                pend_prereqs.remove(comput_idx)  # It must be present.
                fin_prereqs = dt.fin_prereqs
                assert comput_idx not in fin_prereqs
                fin_prereqs.add(comput_idx)

                if len(pend_prereqs) == 0:
                    # Ready to go, make recursion!
                    self._add_term_eval(events, computs, dc_i, dt_i)
                continue

        return

    #
    # Top-level driver functions
    # --------------------------
    #

    def doprint(self, eval_seq, origs=None, separate_decls=False):
        """Make full printing of the evaluation steps.

        This is the main driver function to generate code for tensor
        computations.

        Parameters
        ----------

        eval_seq
            An iterable of tensor definitions for the full evaluation sequence.

        origs
            An iterable of the original tensor definitions before the
            optimization.  When it is not given, no computation is going to be
            considered to be an intermediate.

        separate_decls
            When it is set to true, the declarations and the computations are
            returned as two separate strings.  It is mostly useful for languages
            requiring declarations to be centralized on the top of a scope.
            Not relevant for languages without declarations.

        Notes
        -----

        **For printer writers:**

        Internally, first the computations will be transformed into a linear
        list of events by :py:meth:`form_events`.  Possible events include

        - :py:cls:`TensorDecl`
        - :py:cls:`BeforeComp`
        - :py:cls:`CompTerm`
        - :py:cls:`OutOfUse`

        Individual printers should override the abstract methods

        - :py:meth:`print_decl`
        - :py:meth:`print_before_comp`
        - :py:meth:`print_comp_term`
        - :py:meth:`print_out_of_use`

        to generate code for each of these events.  None can always be returned
        to skip the code generation for a particle event.  This framework can be
        useful for basically all programming languages common in scientific
        computing.

        """
        events = self.form_events(eval_seq, origs)
        decls = []
        execs = []

        dispatch = {
            BeforeComp: self.print_before_comp,
            CompTerm: self.print_comp_term,
            OutOfUse: self.print_out_of_use
        }

        for event in events:
            if isinstance(event, TensorDecl):
                code = self.print_decl(event)
                if code is not None:
                    decls.append(code)
            else:
                cls = type(event)
                if cls not in dispatch:
                    raise ValueError('Invalid event', event)
                code = dispatch[cls](event)
                if code is not None:
                    execs.append(code)
            continue

        if separate_decls:
            return '\n'.join(decls), '\n'.join(execs)
        else:
            return '\n'.join(itertools.chain(decls, execs))

    @abc.abstractmethod
    def print_decl(self, event: TensorDecl) -> typing.Optional[str]:
        """Print the declaration of an intermediate tensor.

        For languages without explicit declaration, this function can simply
        return None.
        """
        pass

    @abc.abstractmethod
    def print_before_comp(self, event: BeforeComp) -> typing.Optional[str]:
        """Print the code before the first computation of an intermediate.

        Normally, the tensor to be computed needs to be initialized to zero
        before its terms are added to it.  Also for some cases, memory
        allocation may need to be performed for intermediates.
        """
        pass

    @abc.abstractmethod
    def print_comp_term(self, event: CompTerm) -> typing.Optional[str]:
        """Print the computation of a tensor term.

        The code should add the term to the target as well.
        """
        pass

    @abc.abstractmethod
    def print_out_of_use(self, event: OutOfUse) -> typing.Optional[str]:
        """Print the code to execute after an intermediate is out-of-use.

        Typically, the memory associated with the intermediate tensor can be
        freed in this event.
        """
        pass


def mangle_base(func):
    """Mangle the base names in the indexed nodes in template context.

    A function taking the printed string for an indexed base and a list of its
    indices, as described in :py:meth:`BasePrinter.transl`, to return a new
    mangled base name can be given to get a function call-back compatible with
    the ``indexed_proc_cb`` argument of :py:meth:`BasePrinter.__init__`
    constructor.

    This function can also be used as a function decorator.  For instance, for a
    tensor with name ``f``, when we have operations on subspaces of the indices
    but the tensor is stored as a whole, we might want to print the base as
    slices depending on the range of the indices given to it.  If we have two
    ranges stored in variables ``o`` and ``v`` and they are over the indices
    ``0:m`` and ``m:n``, the following function::

        @mangle_base
        def print_indexed_base(base, indices):
            o_slice = '0:m'
            v_slice = 'm:n'
            if base == 'f':
                return 'f[{}]'.format(','.join(
                    o_slice if i.range == o else v_slice for i in indices
                ))
            else:
                return base

    can be given to the ``indexed_proc_cb`` argument of
    :py:meth:`BasePrinter.__init__` constructor, so that all appearances of
    ``f`` will be printed as the correct slice depending on the range of the
    indices.  When different slices of ``f`` are actually stored in different
    variables, we can also return the correct variable name inside the function.

    """

    @functools.wraps(func)
    def _mangle_base(node):
        """Mangle the base name according to user-given mangling function."""
        node.base = func(node.base, node.indices)
        return

    return _mangle_base


#
# The imperative code printers
# ----------------------------
#


class ImperativeCodePrinter(BasePrinter):
    """Printer for automatic generation of naive imperative code.

    This printer supports the printing of the evaluation of tensor
    expressions by simple loops and arithmetic operations.

    This is mostly a base class that is going to be subclassed for different
    languages.  For each language, mostly just the options for the language
    could be given in the super initializer.  Most important ones are the
    printer for the scalar expressions and the formatter of loops, as well as
    some definition of literals and operators.

    """

    def __init__(self, scal_printer: Printer, print_indexed_cb,
                 global_indent=1, indent_size=4, max_width=80,
                 line_cont='', breakable_regex=r'(\s*[+-]\s*)', stmt_end='',
                 add_globals=None, add_filters=None,
                 add_tests=None, add_templ=None, **kwargs):
        """
        Initialize the automatic code printer.

        scal_printer
            A sympy printer used for the printing of scalar expressions.

        print_indexed_cb
            It will be called with the printed base, and the list of indices (as
            described in :py:meth:`BasePrinter.transl`) to return the string for
            the printed form.  This will be called after the given processing of
            indexed nodes.

        global_indent
            The base global indentation of the generated code.

        indent_size
            The size of the indentation.

        max_width
            The maximum width for each line.

        line_cont
            The string used for indicating line continuation.

        breakable_regex
            The regular expression used to break long expressions.

        stmt_end
            The ending of the statements.

        index_paren
            The pair of parenthesis for indexing arrays.

        All options to the base class :py:class:`BasePrinter` are also
        supported.

        """

        # Some globals for template rendering.
        default_globals = {
            'global_indent': global_indent,
            'indent_size': indent_size,
            'max_width': max_width,
            'line_cont': line_cont,
            'breakable_regex': breakable_regex,
            'stmt_end': stmt_end,
        }
        if add_globals is not None:
            default_globals.update(add_globals)

        # Initialize the base class.
        super().__init__(
            scal_printer,
            add_globals=default_globals,
            add_filters=add_filters, add_tests=add_tests, add_templ=add_templ,
            **kwargs
        )

        self._print_indexed = print_indexed_cb

    def proc_ctx(
            self, tensor_def: TensorDef, term: typing.Optional[Term],
            tensor_entry: types.SimpleNamespace,
            term_entry: typing.Optional[types.SimpleNamespace]
    ):
        """Process the context.

        The indexed nodes will be printed by user-given printer and given to
        ``indexed`` attributes of the same node.  Also the term contexts will be
        given an attribute named ``amp`` for the whole amplitude part put
        together.
        """

        # This does the processing of the indexed nodes.
        super().proc_ctx(tensor_def, term, tensor_entry, term_entry)

        if term is None:
            tensor_entry.indexed = self._print_indexed(
                tensor_entry.base, tensor_entry.indices
            )
        else:
            factors = []

            if term_entry.numerator != '1':
                factors.append(term_entry.numerator)

            for i in term_entry.indexed_factors:
                i.indexed = self._print_indexed(i.base, i.indices)
                factors.append(i.indexed)
                continue

            factors.extend(term_entry.other_factors)

            parts = [' * '.join(factors)]
            if term_entry.denominator != '1':
                parts.extend(['/', term_entry.denominator])

            term_entry.amp = ' '.join(parts)

        return

    def print_eval(self, ctx: types.SimpleNamespace):
        """Print the evaluation of a tensor definition.
        """
        return self.render('imperative', ctx)


#
# C printer.
#


def print_c_indexed(base, indices):
    """Print indexed objects according to the C syntax.

    The indexed will be printed as multi-dimensional array.
    """
    return ''.join([base] + [
        '[{}]'.format(i.index) for i in indices
    ])


class CPrinter(ImperativeCodePrinter):
    """C code printer.

    In this class, just some parameters for C programming language is fixed
    relative to the base :py:class:`ImperativeCodePrinter`.
    """

    def __init__(self, print_indexed_cb=print_c_indexed, **kwargs):
        """Initialize a C code printer.

        The printer class, the name of the template, the line continuation
        symbol, and the statement ending will be set automatically.
        """

        super().__init__(
            CCodePrinter(),
            print_indexed_cb=print_indexed_cb,
            line_cont='\\', stmt_end=';',
            add_filters={
                'form_loop_beg': _form_c_loop_beg,
                'form_loop_end': _form_c_loop_end,
            }, add_globals={
                'zero_literal': '0.0'
            },
            **kwargs
        )


#
# Some filters for C programming language
#


def _form_c_loop_beg(ctx):
    """Form the loop beginning for C."""
    return 'for({index}={lower}; {index}<{upper}, {index}++)'.format(
        index=ctx.index, lower=ctx.lower, upper=ctx.upper
    ) + ' {'


def _form_c_loop_end(_):
    """Form the loop ending for C."""
    return '}'


#
# Fortran printer.
#


def print_fortran_indexed(base, indices):
    """Print indexed objects according to the Fortran syntax.

    By default, the multi-dimensional array format will be used.
    """
    return base + (
        '' if len(indices) == 0 else '({})'.format(', '.join(
            i.index for i in indices
        ))
    )


class FortranPrinter(ImperativeCodePrinter):
    """Fortran code printer.

    In this class, just some parameters for the *new* Fortran programming
    language is fixed relative to the base :py:class:`ImperativeCodePrinter`.
    """

    def __init__(
            self, openmp=True, print_indexed_cb=print_fortran_indexed,
            default_type='real', heap_interm=True, explicit_bounds=False,
            **kwargs
    ):
        """Initialize a Fortran code printer.

        The printer class, the name of the template, and the line continuation
        symbol will be set automatically.

        Parameters
        ----------

        openmp
            If the evaluation is to be parallelized by OpenMP pragma.

        print_indexed_cb
            The callback to print tensor components.

        default_type
            The default data type for tensor declarations.

        heap_interm
            If intermediates are to be allocated on heap by default.

        explicit_bounds
            If the lower and upper bounds of the tensors are to be explicitly
            written in declarations and allocations.

        """

        if openmp:
            add_templ = {
                'tensor_prelude': _FORTRAN_OMP_PARALLEL_PRELUDE,
                'tensor_finale': _FORTRAN_OMP_PARALLEL_FINALE,
                'init_prelude': _FORTRAN_OMP_INIT_PRELUDE,
                'init_finale': _FORTRAN_OMP_INIT_FINALE,
                'term_prelude': _FORTRAN_OMP_TERM_PRELUDE,
                'term_finale': _FORTRAN_OMP_TERM_FINALE,
            }
        else:
            add_templ = None

        super().__init__(
            FCodePrinter(settings={'source_format': 'free'}),
            print_indexed_cb=print_indexed_cb,
            line_cont='&',
            add_filters={
                'form_loop_beg': self._form_fortran_loop_beg,
                'form_loop_end': self._form_fortran_loop_end,
            }, add_globals={
                'zero_literal': '0.0'
            }, add_templ=add_templ,
            **kwargs
        )

        self._default_type = default_type
        self._heap_interm = heap_interm
        self._explicit_bounds = explicit_bounds

        base_indent_size = int(self._env.globals['global_indent']) * int(
            self._env.globals['indent_size']
        )
        self._base_indent = ' ' * base_indent_size

    def doprint(self, eval_seq, origs=None, separate_decls=False):
        """Make full printing of the evaluation steps.
        """
        events = self.form_events(eval_seq, origs)
        decls = []
        execs = []

        for event in events:
            if isinstance(event, _TensorDecl):
                decls.append(self.print_decl(
                    event.comput.ctx, allocatable=self._heap_interm
                ))
            elif isinstance(event, _BeforeCompute):
                comput = event.comput
                ctx = comput.ctx
                if self._heap_interm and comput.is_interm and len(
                        ctx.indices
                ) > 0:
                    execs.append(self.print_alloc(ctx))
            elif isinstance(event, _TensorComput):
                execs.append(self.print_eval(event.ctx))
            elif isinstance(event, _NoLongerInUse):
                ctx = event.comput.ctx
                if self._heap_interm and len(ctx.indices) > 0:
                    execs.append(self.print_dealloc(ctx))
            else:
                assert False  # Unknown event.

        if separate_decls:
            return '\n'.join(decls), '\n'.join(execs)
        else:
            return '\n'.join(itertools.chain(decls, execs))

    def print_decl_eval(
            self, tensor_defs: typing.Iterable[TensorDef],
            decl_type=None, explicit_bounds=None
    ) -> typing.Tuple[typing.List[str], typing.List[str]]:
        """Print Fortran declarations and evaluations of tensor definitions.

        Parameters
        ----------

        tensor_defs
            The tensor definitions to print.

        decl_type
            The type to be declared for the tensors.  By default, the value set
            for the printer will be used.

        explicit_bounds
            If the lower and upper bounds should be written explicitly in the
            declaration.  By default, the value set for the printer will be
            used.

        Return
        ------

        decls
            The list of declaration strings.

        evals
            The list of evaluation strings.

        """

        if decl_type is None:
            decl_type = self._default_type
        if explicit_bounds is None:
            explicit_bounds = self._explicit_bounds

        decls = []
        evals = []

        for tensor_def in tensor_defs:
            ctx = self.transl(tensor_def)
            decls.append(self.print_decl(ctx, decl_type, explicit_bounds))
            evals.append(self.print_eval(ctx))
            continue

        return decls, evals

    def print_decl(
            self, ctx, decl_type=None, explicit_bounds=None, allocatable=False
    ):
        """Print the Fortran declaration of the LHS of a tensor definition.

        A string will be returned that forms the naive declaration of the
        given tensor as local variables.

        """

        decl_type = self._default_type if decl_type is None else decl_type
        explicit_bounds = (
            self._explicit_bounds if explicit_bounds is None else
            explicit_bounds
        )

        if len(ctx.indices) > 0:
            if allocatable:
                bounds = ', '.join(':' for _ in ctx.indices)
            else:
                bounds = self._form_bounds(ctx, explicit_bounds)
            sizes_decl = ', dimension({})'.format(bounds)
            if allocatable:
                sizes_decl += ', allocatable'
        else:
            sizes_decl = ''

        return ''.join([
            self._base_indent, decl_type, sizes_decl, ' :: ', ctx.base
        ])

    def print_alloc(self, ctx, explicit_bounds=None):
        """Print the allocation statement.
        """
        explicit_bounds = (
            self._explicit_bounds if explicit_bounds is None else
            explicit_bounds
        )
        bounds = self._form_bounds(ctx, explicit_bounds)
        return ''.join([
            self._base_indent, 'allocate(', ctx.base, '(', bounds, '))'
        ])

    def print_dealloc(self, ctx):
        """Print the deallocation command.
        """
        return ''.join([
            self._base_indent, 'deallocate(', ctx.base, ')'
        ])

    def _form_bounds(self, ctx, explicit_bounds):
        """Form the string for array bounds.
        """
        return ', '.join(
            ':'.join([self._print_lower(i.lower_expr), i.upper])
            if explicit_bounds else i.size
            for i in ctx.indices
        )

    def _print_lower(self, lower: Expr):
        """Print the lower bound based on the Fortran convention.
        """
        return self._print_scal(lower + Integer(1))

    def _form_fortran_loop_beg(self, ctx):
        """Form the loop beginning for Fortran."""

        lower = self._print_lower(ctx.lower_expr)

        return 'do {index}={lower}, {upper}'.format(
            index=ctx.index, lower=lower, upper=ctx.upper
        )

    @staticmethod
    def _form_fortran_loop_end(_):
        """Form the loop ending for Fortran."""
        return 'end do'


_FORTRAN_OMP_PARALLEL_PRELUDE = """\
!$omp parallel default(shared)
"""

_FORTRAN_OMP_PARALLEL_FINALE = "!$omp end parallel\n"

_FORTRAN_OMP_INIT_PRELUDE = """\
{% if n_ext > 0 %}
!$omp do schedule(static)
{% else %}
!$omp single
{% endif %}
"""
_FORTRAN_OMP_INIT_FINALE = """\
{% if n_ext > 0 %}
!$omp end do
{% else %}
!$omp end single
{% endif %}
"""

_FORTRAN_OMP_TERM_PRELUDE = """\
{% if n_ext > 0 %}
!$omp do schedule(static)
{% else %}
{% if (term.sums | length) > 0 %}
!$omp do schedule(static) reduction(+:{{ lhs }})
{% else %}
!$omp single
{% endif %}
{% endif %}
"""

_FORTRAN_OMP_TERM_FINALE = """\
{% if (n_ext + (term.sums | length)) > 0 %}
!$omp end do
{% else %}
!$omp end single
{% endif %}
"""


#
# Einsum printer
# --------------
#


class EinsumPrinter(BasePrinter):
    """Printer for the einsum function.

    For tensors that are classical tensor contractions, this printer generates
    code based on the NumPy ``einsum`` function.  For contractions supported,
    the code from this printer can also be used for Tensorflow.

    """

    def __init__(self, **kwargs):
        """Initialize the printer.

        All keyword arguments are forwarded to the base class
        :py:class:`BasePrinter`.
        """

        super().__init__(PythonPrinter(), **kwargs)

    def print_eval(
            self, tensor_defs: typing.Iterable[TensorDef],
            base_indent=4
    ) -> str:
        """Print the evaluation of the tensor definitions.

        Parameters
        ----------

        tensor_defs
            The tensor definitions for the evaluations.

        base_indent
            The base indent of the generated code.

        Return
        ------

        The code for evaluations.

        """

        ctxs = []
        for tensor_def in tensor_defs:
            ctx = self.transl(tensor_def)
            for i in ctx.terms:

                for j in i.other_factors_expr:
                    indices = []

                    def _replace_indexed(*args):
                        """Replace indexed quantity in expression."""
                        indices.append(args[1:])
                        return args[0].args[0]

                    repled = j.replace(Indexed, _replace_indexed)
                    if len(indices) > 1:
                        raise ValueError(
                            'Expression too complicated for einsum', j
                        )

                    indices = indices[0]
                    factor_ctx = types.SimpleNamespace()
                    factor_ctx.base = self._print_scal(repled)
                    # Einsum does not really depend on the ranges.
                    factor_ctx.indices = self._form_indices_ctx((
                        (i, None) for i in indices
                    ), enforce=False)
                    i.indexed_factors.append(factor_ctx)

                continue

            ctxs.append(ctx)
            continue

        code = '\n'.join(
            self.render('einsum', i) for i in ctxs
        )

        return textwrap.indent(code, ' ' * base_indent)
