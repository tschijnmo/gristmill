"""Tests for the base printer.
"""

import subprocess
import textwrap
from unittest.mock import patch

import pytest
from sympy import Symbol, IndexedBase, symbols
from sympy.printing.python import PythonPrinter

from drudge import Drudge, Range
from gristmill import BasePrinter, FortranPrinter, EinsumPrinter, mangle_base
from gristmill.generate import (
    _TensorDecl, _BeforeCompute, _ComputeTerm, _NoLongerInUse
)


@pytest.fixture(scope='module')
def simple_drudge(spark_ctx):
    """Form a simple drudge with some basic information.
    """

    dr = Drudge(spark_ctx)

    n = Symbol('n')
    r = Range('R', 0, n)

    dumms = symbols('a b c d e f g')
    dr.set_dumms(r, dumms)
    dr.add_resolver_for_dumms()

    return dr


@pytest.fixture
def colourful_tensor(simple_drudge):
    """Form a colourful tensor definition capable of large code coverage.
    """

    dr = simple_drudge
    p = dr.names

    x = IndexedBase('x')
    u = IndexedBase('u')
    v = IndexedBase('v')
    dr.set_name(x, u, v)

    r, s = symbols('r s')
    dr.set_name(r, s)

    a, b, c = p.R_dumms[:3]

    tensor = dr.define(x[a, b], (
            ((2 * r) / (3 * s)) * u[b, a] -
            dr.sum((c, p.R), u[a, c] * v[c, b] * c ** 2 / 2)
    ))

    return tensor


@pytest.fixture
def eval_seq_deps(simple_drudge):
    """A simple evaluation sequence with some dependencies.

    Here, the tensors are all matrices. we have inputs X, Y.

    I1 = X Y
    I2 = Y X
    I3 = Tr(I1)

    R1 = I1 * I3 + I2
    R2 = I1 * 2

    """

    dr = simple_drudge
    p = dr.names
    a, b, c = p.a, p.b, p.c

    x = IndexedBase('X')
    y = IndexedBase('Y')
    i1 = IndexedBase('I1')
    i2 = IndexedBase('I2')
    i3 = Symbol('I3')
    r1 = IndexedBase('R1')
    r2 = IndexedBase('R2')

    i1_def = dr.define_einst(i1[a, b], x[a, c] * y[c, b])
    i2_def = dr.define_einst(i2[a, b], y[a, c] * x[c, b])
    i3_def = dr.define_einst(i3, i1[a, a])
    r1_def = dr.define_einst(r1[a, b], i1[a, b] * i3 + i2[a, b])
    r2_def = dr.define_einst(r2[a, b], i1[a, b] * 2)

    return [i1_def, i2_def, i3_def, r1_def, r2_def], [r1_def, r2_def]


def test_base_printer_ctx(simple_drudge, colourful_tensor):
    """Test the context formation facility in base printer."""

    dr = simple_drudge
    p = dr.names
    tensor = colourful_tensor

    # Process indexed names by mangling the base name.
    with patch.object(BasePrinter, '__abstractmethods__', frozenset()):
        printer = BasePrinter(PythonPrinter(), mangle_base(
            lambda base, indices: base + str(len(indices))
        ))
    ctx = printer.transl(tensor)

    def check_range(ctx, index):
        """Check the range information in a context for a index."""
        assert ctx.index == index
        assert ctx.range == p.R
        assert ctx.lower == '0'
        assert ctx.upper == 'n'
        assert ctx.size == 'n'

    assert ctx.base == 'x2'
    for i, j in zip(ctx.indices, ['a', 'b']):
        check_range(i, j)
        continue

    assert len(ctx.terms) == 2
    for term in ctx.terms:
        if len(term.sums) == 0:
            # The transpose term.

            assert term.phase == '+'
            assert term.numerator == '2*r'
            assert term.denominator == '(3*s)'

            assert len(term.indexed_factors) == 1
            factor = term.indexed_factors[0]
            assert factor.base == 'u2'
            for i, j in zip(factor.indices, ['b', 'a']):
                check_range(i, j)
                continue

            assert len(term.other_factors) == 0

        elif len(term.sums) == 1:

            check_range(term.sums[0], 'c')

            assert term.phase == '-'
            assert term.numerator == '1'
            assert term.denominator == '2'

            assert len(term.indexed_factors) == 2
            for factor in term.indexed_factors:
                if factor.base == 'u2':
                    expected = ['a', 'c']
                elif factor.base == 'v2':
                    expected = ['c', 'b']
                else:
                    assert False
                for i, j in zip(factor.indices, expected):
                    check_range(i, j)
                    continue
                continue

            assert len(term.other_factors) == 1
            assert term.other_factors[0] == 'c**2'

        else:
            assert False


def test_events_generation(eval_seq_deps):
    """Test the event generation facility in the base printer."""
    eval_seq, origs = eval_seq_deps

    with patch.object(BasePrinter, '__abstractmethods__', frozenset()):
        printer = BasePrinter(PythonPrinter())
    events = printer.form_events(eval_seq, origs)

    i1 = IndexedBase('I1')
    i2 = IndexedBase('I2')
    i3 = Symbol('I3')
    r1 = IndexedBase('R1')
    r2 = IndexedBase('R2')

    events.reverse()  # For easy popping from front.

    for i in [i1, i2, i3]:
        event = events.pop()
        assert isinstance(event, _TensorDecl)
        assert event.comput.target == i
        continue

    event = events.pop()
    assert isinstance(event, _BeforeCompute)
    assert event.comput.target == i1

    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == i1
    assert event.term_idx == 0

    # I1 drives I3.
    event = events.pop()
    assert isinstance(event, _BeforeCompute)
    assert event.comput.target == i3

    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == i3
    assert event.term_idx == 0

    # I3, I1, drives the first term of R1.
    event = events.pop()
    assert isinstance(event, _BeforeCompute)
    assert event.comput.target == r1

    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == r1
    assert event.term_idx == 0

    # Now I3 should be out of dependency.
    event = events.pop()
    assert isinstance(event, _NoLongerInUse)
    assert event.comput.target == i3

    # Another one driven by I1.
    event = events.pop()
    assert isinstance(event, _BeforeCompute)
    assert event.comput.target == r2

    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == r2
    assert event.term_idx == 0

    # I1 no longer needed any more.
    event = events.pop()
    assert isinstance(event, _NoLongerInUse)
    assert event.comput.target == i1

    # Nothing driven.
    event = events.pop()
    assert isinstance(event, _BeforeCompute)
    assert event.comput.target == i2

    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == i2
    assert event.term_idx == 0

    # The last term in R1.
    event = events.pop()
    assert isinstance(event, _ComputeTerm)
    assert event.comput.target == r1
    assert event.term_idx == 1

    # Finally, free I2.
    event = events.pop()
    assert isinstance(event, _NoLongerInUse)
    assert event.comput.target == i2

    assert len(events) == 0


def _test_fortran_code(code, dir):
    """Test the given Fortran code in the given directory.

    The Fortran code is expected to generate an output of ``OK``.
    """

    orig_cwd = dir.chdir()

    dir.join('test.f90').write(code)
    stat = subprocess.run(['gfortran', '-o', 'test', '-fopenmp', 'test.f90'])
    assert stat.returncode == 0
    stat = subprocess.run(['./test'], stdout=subprocess.PIPE)
    assert stat.stdout.decode().strip() == 'OK'

    orig_cwd.chdir()
    return True


def test_basic_fortran_printer(colourful_tensor, tmpdir):
    """Test the basic functionality of the Fortran printer."""

    tensor = colourful_tensor

    printer = FortranPrinter()
    decls, evals = printer.print_decl_eval([tensor])
    assert len(decls) == 1
    assert len(evals) == 1

    code = _FORTRAN_BASIC_TEST_CODE.format(decl=decls[0], eval=evals[0])
    assert _test_fortran_code(code, tmpdir)


_FORTRAN_BASIC_TEST_CODE = """
program main
implicit none

integer, parameter :: n = 100
real :: r = 6
real :: s = 2
integer :: a, b, c

real, dimension(n, n) :: u
real, dimension(n, n) :: v

{decl}
real, dimension(n, n) :: diag
real, dimension(n, n) :: expected

call random_number(u)
call random_number(v)

{eval}

diag = 0
do a = 1, n
    diag(a, a) = real(a ** 2) / 2
end do

expected = transpose(u) * 2 * r / (3 * s)
expected = expected - matmul(u, matmul(diag, v))

if (any(abs(x - expected) / abs(expected) > 1.0E-5)) then
    write(*, *) "WRONG"
end if

write(*, *) "OK"

end program main
"""


def test_full_fortran_printer(eval_seq_deps, tmpdir):
    """Test the Fortran printer for full evaluation."""

    eval_seq, origs = eval_seq_deps

    printer = FortranPrinter(openmp=False)
    evals = printer.doprint(eval_seq, origs)

    code = _FORTRAN_FULL_TEST_CODE.format(eval=evals)
    assert _test_fortran_code(code, tmpdir)

    sep_code = printer.doprint(
        eval_seq, origs, separate_decls=True
    )
    assert len(sep_code) == 2
    assert evals == '\n'.join(sep_code)


_FORTRAN_FULL_TEST_CODE = """
program main
implicit none

integer, parameter :: n = 10
integer :: a, b, c

real, dimension(n, n) :: x
real, dimension(n, n) :: y
real, dimension(n, n) :: r1
real, dimension(n, n) :: r2
real, dimension(n, n) :: expected_r1
real, dimension(n, n) :: expected_r2

call random_number(x)
call random_number(y)

block
{eval}
end block

block
real, dimension(n, n) :: xy
real :: trace

xy = matmul(x, y)

trace = 0
do a = 1, n
    trace = trace + xy(a, a)
end do

expected_r1 = xy * trace
expected_r2 = xy * 2

end block

if (any(abs(r1 - expected_r1) / abs(expected_r1) > 1.0E-5)) then
    write(*, *) "WRONG"
end if
if (any(abs(r2 - expected_r2) / abs(expected_r2) > 1.0E-5)) then
    write(*, *) "WRONG"
end if

write(*, *) "OK"

end program main
"""


def test_einsum_printer(simple_drudge):
    """Test the functionality of the einsum printer."""

    dr = simple_drudge
    p = dr.names
    a, b, c = p.R_dumms[:3]

    x = IndexedBase('x')
    u = IndexedBase('u')
    v = IndexedBase('v')

    tensor = dr.define_einst(
        x[a, b], u[b, a] ** 2 - 2 * u[a, c] * v[c, b] / 3
    )

    printer = EinsumPrinter()
    code = printer.print_eval([tensor])

    for line in code.splitlines():
        assert line[:4] == ' ' * 4
        continue

    exec_code = _EINSUM_DRIVER_CODE.format(code=textwrap.dedent(code))
    env = {}
    exec(exec_code, env, {})
    assert env['diff'] < 1.0E-5  # Arbitrary delta.


_EINSUM_DRIVER_CODE = """
from numpy import einsum, array
from numpy import linalg

u = array([[1.0, 2], [3, 4]])
v = array([[1.0, 0], [0, 1]])

{code}

expected = (u ** 2).transpose() - (2.0 / 3) * u @ v
global diff
diff = linalg.norm(x - expected)

"""
