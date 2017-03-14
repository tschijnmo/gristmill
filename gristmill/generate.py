"""Generate source code from optimized computations."""

import types
import typing

from drudge import TensorDef, Term, Range, prod_
from drudge.term import try_resolve_range
from sympy import (
    Expr, Mul, Pow, Integer, Rational, Add, Indexed, IndexedBase
)
from sympy.printing.ccode import CCodePrinter
from sympy.printing.fcode import FCodePrinter
from sympy.printing.printer import Printer

from .utils import create_jinja_env


class BasePrinter:
    """The base class for tensor printers.
    """

    def __init__(self, scal_printer: Printer, indexed_proc_cb=lambda x: None,
                 add_globals=None, add_filters=None, add_tests=None,
                 add_templ=None):
        """Initializes a base printer.

        Parameters
        ----------

        scal_printer
            The SymPy printer for scalar quantities.

        indexed_proc_cb
            It is going to be called with context nodes with ``base`` and
            ``indices`` (in both the root and for each indexed factors, as
            described in :py:meth:`transl`) to do additional processing.

        """

        env = create_jinja_env(add_filters, add_globals, add_tests, add_templ)

        self._env = env
        self._scal_printer = scal_printer
        self._indexed_proc = indexed_proc_cb

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
            range it is over. For convenience, ``lower``, ``upper``, and
            ``size`` have the printed form of lower/upper bounds and the size of
            the range.

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

        The actual content of the context can also be customized by overriding
        the :py:meth:`proc_ctx` in subclasses.

        """

        ctx = types.SimpleNamespace()

        base = tensor_def.base
        ctx.base = self._print_scal(
            base.label if isinstance(base, IndexedBase) else base
        )
        ctx.indices = self._form_indices_ctx(tensor_def.exts)

        # The stack keeping track of the external and internal indices for range
        # resolution.
        indices_dict = dict(tensor_def.exts)
        resolvers = tensor_def.rhs.drudge.resolvers.value

        terms = []
        ctx.terms = terms

        # Render each term in turn.
        for term in tensor_def.rhs_terms:

            term_ctx = types.SimpleNamespace()
            terms.append(term_ctx)

            indices_dict.update(term.sums)
            term_ctx.sums = self._form_indices_ctx(term.sums)

            factors, coeff = term.amp_factors

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
        return templ.render(ctx)

    def _form_indices_ctx(
            self, pairs: typing.Iterable[typing.Tuple[Expr, Range]],
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
            else:
                lower = self._print_scal(range_.lower)
                upper = self._print_scal(range_.upper)
                size = self._print_scal(range_.size)

            res.append(types.SimpleNamespace(
                index=self._print_scal(index),
                range=range_, lower=lower, upper=upper, size=size
            ))
            continue

        return res

    def _print_scal(self, expr: Expr):
        """Print a scalar."""
        return self._scal_printer.doprint(expr)


