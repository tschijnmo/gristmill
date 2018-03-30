"""General utilities."""

import functools
import operator
import re
import typing

import numpy as np
from jinja2 import (
    Environment, PackageLoader, ChoiceLoader, DictLoader
)
from numpy.polynomial import Polynomial
from sympy import Symbol, Integer, Mul, poly_from_expr, Number, Poly, Expr

from drudge import prod_, TensorDef, Range


#
# Cost-related utilities
# ----------------------
#
# Numeric cost manipulation during optimization.
#

class SVPoly(Polynomial):
    """Single variate polynomials for sizes and costs.

    The primary thing added to its numpy base class is the ordering.  But this
    ordering has caveats.  Only when comparing with **integer zero**, the
    leading coefficient and the possible presence of infinity will be checked to
    see if the size is an asymptotically positive/negative one.  In all other
    situations, non-negative size will be assumed.  The comparison is going to
    be based on degree of the polynomial and the lexicographical order of the
    coefficients.

    """

    def __lt__(self, other):
        """Make a less than comparison."""
        return self._comp(other) < 0

    def __gt__(self, other):
        """Make a greater than comparison."""
        return self._comp(other) > 0

    def __eq__(self, other):
        """Make an equality comparison."""
        return self._comp(other) == 0

    def __ge__(self, other):
        """Make a greater than or equal to comparison."""
        return self > other or self == other

    def _comp(self, other):
        """Make a comparison with another size quantity."""

        if other is 0:
            return self._comp_w_zero()

        l_deg = self.degree()
        r_deg = other.degree() if isinstance(other, SVPoly) else 0

        if l_deg < r_deg:
            return -1
        elif l_deg > r_deg:
            return 1
        else:
            diff = self - other
            return diff.coef[-1]

    def _comp_w_zero(self):
        """Test if a cost is a positive/negative one."""

        coeff = self.coef
        inf_idxes, = np.where(np.isinf(coeff))
        if inf_idxes.size == 0:
            idx = -1
        else:
            idx = inf_idxes[-1]
        return coeff[idx]


# Type for sizes, or costs, especially for annotation.
#
# Primarily, we only use addition, subtraction, multiplication, and order
# comparison.

Size = typing.Union[int, float, SVPoly]


def form_size(expr: Expr) -> typing.Tuple[Size, typing.Optional[Symbol]]:
    """Form a size object from a SymPy expression."""

    symbs = expr.atoms(Symbol)
    n_symbs = len(symbs)

    if n_symbs == 0:
        symb = None
        coeff_exprs = [expr]
    elif n_symbs == 1:
        symb = symbs.pop()
        coeff_exprs = Poly(expr, symb).all_coeffs()
        coeff_exprs.reverse()
    else:
        raise ValueError(
            'Invalid expression', expr,
            'expecting single variate polynomial (or number)'
        )

    if all(i.is_integer for i in coeff_exprs):
        dtype = int
    else:
        dtype = float

    if len(coeff_exprs) > 1:
        coeffs = np.array(coeff_exprs, dtype=dtype)
        cost = SVPoly(coeffs)
    elif len(coeff_exprs) == 1:
        cost = dtype(coeff_exprs[0])
    else:
        assert False

    return cost, symb


def mul_sizes(sizes):
    """Multiply sizes in an iterable together.

    The multiplication is going to be based on integer unity, with the actual
    type of the result determined by the result of the multiplication
    operations.
    """
    return functools.reduce(operator.mul, sizes, 1)


def get_total_size(sums) -> Size:
    """Get the total size of a summation list.

    Here an integral unity will be returned when we have an empty summation
    list, or we shall have the product of the sizes of the ranges.
    """

    size = 1
    for _, i in sums:
        curr = i.size
        if curr is None:
            raise ValueError(
                'Invalid range for optimization', i,
                'expecting a bound range.'
            )
        size *= curr
        continue

    return size


class SizedRange(Range):
    """Ranges with polynomial sizes.

    This subclass has the size of the ranges in NumPy polynomial form cached to
    avoid repeated computation.  Note that the explicit bounds are dropped for
    faster equality and hashing.
    """

    __slots__ = [
        '_size'
    ]

    def __init__(self, label, size):
        """Initialize the sized range object."""
        super().__init__(label)
        self._size = size

    @property
    def size(self):
        """Get the size of the range."""
        return self._size

    def replace_label(self, new_label):
        """Replace the label of the range."""
        return SizedRange(new_label, self._size)

    @property
    def sort_key(self):
        """The sort key for the sized range.

        This override ensures that the ranges are always sorted by increasing
        size after the canonicalization.
        """
        return (self._size, self._label)


def form_sized_range(range_: Range, substs) -> typing.Tuple[
    SizedRange, typing.Optional[Symbol]
]:
    """Form a sized range from the original raw range.

    The when a symbol exists in the ranges, it will be returned as the second
    result, or the second result will be none.
    """

    if not range_.bounded:
        raise ValueError(
            'Invalid range for optimization', range_,
            'expecting explicit bound'
        )
    lower, upper = [
        i.xreplace(substs)
        for i in [range_.lower, range_.upper]
    ]
    size_expr = upper - lower

    size, symb = form_size(size_expr)

    return SizedRange(range_.label, size), symb


@functools.total_ordering
class Tuple4Cmp(tuple):
    """Simple tuple for comparison.

    Everything is the same as the built-in tuple class, just the equality and
    ordering is solely based on the first item.

    Note that this class does not make any advanced checking of the validity of
    the comparison.
    """

    def __eq__(self, other):
        """Make equality comparison."""
        return self[0] == other[0]

    def __lt__(self, other):
        """Make less-than comparison."""
        return self[0] < other[0]


#
# Public symbolic cost computation function.
#


def get_flop_cost(
        eval_seq: typing.Iterable[TensorDef], leading=False,
        ignore_consts=True
):
    """Get the FLOP cost for the given evaluation sequence.

    This function gives the count of floating-point operations, addition and
    multiplication, involved by the evaluation sequence.  Note that the cost of
    copying and initialization are not counted.  And this function is only
    applicable where the amplitude of the terms are simple products.

    Parameters
    ----------

    eval_seq
        The evaluation sequence whose FLOP cost is to be estimated.  It should
        be given as an iterable of tensor definitions.

    leading
        If only the cost terms with leading scaling be given.  When multiple
        symbols are present in the range sizes, terms with the highest total
        scaling is going to be picked.

    ignore_consts
        If the cost of scaling with constants can be ignored.  :math:`2 x_i y_j`
        could count as just one FLOP when it is set, otherwise it would be two.

    """

    cost = sum(_get_flop_cost(i, ignore_consts) for i in eval_seq)
    return _get_leading(cost) if leading else cost


def _get_flop_cost(step, ignore_consts):
    """Get the FLOP cost of a tensor evaluation step."""

    ext_size = get_total_size(step.exts)

    cost = Integer(0)
    n_terms = 0
    for term in step.rhs_terms:
        sum_size = get_total_size(term.sums)

        if isinstance(term.amp, Mul):
            factors = term.amp.args
        else:
            factors = (term.amp,)

        if ignore_consts:
            factors = (i for i in factors if not isinstance(i, Number))
        else:
            # Minus one should be implemented via subtraction, hence renders no
            # multiplication cost.
            factors = (i for i in factors if abs(i) != 1)

        n_factors = sum(1 for i in factors if abs(i) != 1)
        n_mult = n_factors - 1 if n_factors > 0 else 0

        if sum_size == 1:
            n_add = 0
        else:
            n_add = 1

        cost += (n_add + n_mult) * ext_size * sum_size
        n_terms += 1
        continue

    if n_terms > 1:
        cost += (n_terms - 1) * ext_size

    return cost


def _get_leading(cost):
    """Get the leading terms in a cost polynomial."""

    if cost == 0:
        return cost

    symbs = tuple(cost.atoms(Symbol))
    poly, _ = poly_from_expr(cost, *symbs)
    terms = poly.terms()

    leading_deg = max(sum(i) for i, _ in terms)
    leading_cost = sum(
        coeff * prod_(i ** j for i, j in zip(symbs, degs))
        for degs, coeff in terms
        if sum(degs) == leading_deg
    )

    return leading_cost


#
# Disjoint set forest
# -------------------
#

class DSF(object):
    """
    Disjoint sets forest.

    This is a very simple implementation of the disjoint set forest data
    structure for finding the inseparable chunks of factors for given ranges to
    keep.  Heuristics of union by rank and path compression are both applied.

    """

    def __init__(self, n_elems):
        """Initialize the object.

        Parameters
        ----------

        n_elems

            The number of elements in the forest.
        """

        self._n_elems = n_elems
        self._ranks = [0] * n_elems
        self._parents = list(range(n_elems))
        self._n_sets = n_elems

    def union(self, contents):
        """Put the given elements into union.

        Parameters
        ----------

        contents
            An iterable of the element indices that are going to be put into the
            same set.

        """

        represent = None  # The set that other sets will be unioned to.
        for i in contents:

            if represent is None:
                represent = i
            else:
                self.union_two(represent, i)

            continue

        return None

    def __iter__(self):
        """Iterate over the indices of all points in the forest."""
        return iter(range(self._n_elems))

    @property
    def n_sets(self):
        """The number of sets in the forest.
        """
        return self._n_sets

    def union_two(self, idx1, idx2):
        """Union the two subsets that the two given indices are in."""

        set1 = self.find(idx1)
        set2 = self.find(idx2)

        if set1 == set2:
            return

        self._n_sets -= 1

        rank1 = self._ranks[set1]
        rank2 = self._ranks[set2]

        # Union by rank.
        if rank1 > rank2:
            self._parents[set2] = set1
        else:
            self._parents[set1] = set2
            if rank1 == rank2:
                self._ranks[set2] += 1

    def find(self, idx):
        """Find the root index of the subset the given index is in."""

        parent = self._parents[idx]
        if idx != parent:
            self._parents[idx] = self.find(parent)
        return self._parents[idx]


#
# Jinja environment creation
# --------------------------
#


class JinjaEnv(Environment):
    """A Jinja environment for template rendering.

    This subclass of the normal Jinja environment is designed to be suitable for
    rendering code for tensor computations.  Notably the templates will be
    retrieved from the ``templates`` directory in the gristmill package.  And
    some filters and predicates will be added, including

    - :py:meth:`wrap_line`
    - :py:meth:`form_indent`
    - :py:meth:`non_empty`

    which can be called inside Jinja environments or both as a method.  These
    functionality are the ones that can be useful cross different printers.

    Parameters
    ----------

    indent
        The string used for indentation.  By default to four spaces.

    base_indent
        The base level of indentation for the base level.

    breakable_regex
        The regular expression giving the places where the line can be broke.
        The parts in the regular expression that needs to be kept can be put in
        a capturing parentheses.  It can be set to None to disable automatic
        line breaking.

    max_width
        The maximum width of the lines to wrap the given line within.

    line_cont
        The string to be put by the end of line to indicate line continuation.

    cont_indent
        The number of additional indentation for the continued lines.

    add_filters
        Additional filters to add to the environment.

    add_globals
        Additional globals to add to the environment.

    add_templ
        Additional templates to add to the environment, specially to be used
        inside Jinja templates by the ``include`` directive.

    """

    def __init__(
            self, indent=' ' * 4, base_indent=0,
            breakable_regex=None, max_width=80, line_cont='', cont_indent=1,
            add_filters=None, add_globals=None, add_tests=None, add_templ=None
    ):
        """Initialize the Jinja environment."""

        # Set the Jinja environment up.
        super().__init__(
            trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
            loader=ChoiceLoader(
                [PackageLoader('gristmill')] +
                ([DictLoader(add_templ)] if add_templ is not None else [])
            )
        )

        self._indent = indent
        self._base_indent = base_indent
        self._breakable_regex = breakable_regex
        self._max_width = max_width
        self._line_cont = line_cont
        self._cont_indent = cont_indent

        # Add the default filters and tests for all printers.
        self.globals['indent'] = indent
        self.globals['base_indent'] = indent * base_indent
        self.globals['line_cont'] = line_cont
        self.globals['cont_indent'] = indent * cont_indent

        self.filters['form_indent'] = self.form_indent
        self.filters['wrap_line'] = self.wrap_line

        self.tests['non_empty'] = self.non_empty

        # Add the additional globals, filters, and tests.
        if add_globals is not None:
            self.globals.update(add_globals)
        if add_filters is not None:
            self.filters.update(add_filters)
        if add_tests is not None:
            self.tests.update(add_tests)

    def form_indent(self, level: int, add_base=True) -> str:
        """Form an indentation space block.

        Parameters
        ----------

        level
            The level of the indentation. The content of the indentation is
            going to be the one set for the environment.

        add_base
            If base indent is to be added.

        """

        return self._indent * (
                (self._base_indent if add_base else 0) + level
        )

    def wrap_line(self, line, level):
        """Wrap the given line within the given width.

        This function is also going to be exported to be used by template
        writers in Jinja as a filter.

        Parameters
        ----------

        line
            The line to be wrapped.

        level
            The level for indentation for the line.

        """

        if self._breakable_regex is None:
            return line

        first_indent = self.form_indent(level)
        cont_indent = self.form_indent(level + self._cont_indent)
        max_width = self._max_width
        line_cont = self._line_cont

        # Break the given line according to the given regular expression.  The
        # token may not necessarily be the actual token in the language.
        tokens = re.split(self._breakable_regex, line)

        # Actually break the list of trunks into lines.
        lines = []
        curr_line = [first_indent]
        curr_len = len(first_indent)
        n_tokens = len(tokens)
        for idx, token in enumerate(tokens):
            new_len = curr_len + len(token)
            if_add = (
                    len(curr_line) == 0
                    or (idx + 1 == n_tokens and new_len <= max_width)
                    or new_len + len(line_cont) <= max_width
            )
            if if_add:
                curr_line.append(token)
                curr_len = new_len
            else:
                # When the current line is already filled up.
                curr_line.append(line_cont)
                lines.append(''.join(curr_line))
                curr_line = [cont_indent]
                curr_len = len(cont_indent)
            # Go on to the next token.
            continue

        # We need to add the trailing current line after all the loop.
        lines.append(''.join(curr_line))

        return '\n'.join(lines)

    @staticmethod
    def non_empty(sequence):
        """Test if a given sequence is non-empty."""
        return len(sequence) > 0

    def indent_lines(self, lines: str, level, add_base=True):
        """Indent the lines in the given string.

        This is mostly for usage inside Python script.  For the problem of base
        indentation, we can either use :py:meth:`form_indent` for each line in
        the template when it is convenient to do so, or can be just form
        relative indentation and use this method to decorate all the lines from
        the template.

        """

        indent = self.form_indent(level, add_base=add_base)
        return '\n'.join(
            indent + i for i in lines.splitlines()
        )
