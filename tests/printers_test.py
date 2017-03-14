"""Tests for the base printer.
"""

import subprocess
import textwrap

import pytest
from drudge import Drudge, Range
from sympy import Symbol, IndexedBase, symbols
from sympy.printing.python import PythonPrinter

from gristmill import BasePrinter, FortranPrinter, EinsumPrinter


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


def test_base_printer_ctx(simple_drudge, colourful_tensor):
    """Test the context formation facility in base printer."""

    dr = simple_drudge
    p = dr.names
    tensor = colourful_tensor

    def proc_indexed(ctx):
        """Process indexed names by mangling the base name."""
        ctx.base = ctx.base + str(len(ctx.indices))
        return

    printer = BasePrinter(PythonPrinter(), proc_indexed)
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


def test_fortran_printer(simple_drudge, colourful_tensor, tmpdir):
    """Test the functionality of the Fortran printer."""

    dr = simple_drudge
    tensor = colourful_tensor

    printer = FortranPrinter()
    decls, evals = printer.print_decl_eval([tensor])
    assert len(decls) == 1
    assert len(evals) == 1

    code = _FORTRAN_TEST_CODE.format(decl=decls[0], eval=evals[0])

    orig_cwd = tmpdir.chdir()

    tmpdir.join('test.f90').write(code)
    stat = subprocess.run(['gfortran', '-o', 'test', '-fopenmp', 'test.f90'])
    assert stat.returncode == 0
    stat = subprocess.run(['./test'], stdout=subprocess.PIPE)
    assert stat.stdout.decode().strip() == 'OK'

    orig_cwd.chdir()


_FORTRAN_TEST_CODE = """
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

do a = 1, n
    do b = 1, n
        if (abs(x(a, b) - expected(a, b)) / abs(x(a, b)) > 1.0E-5) then
            write(*, *) "WRONG"
        end if
    end do
end do

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
        x[a, b], u[b, a] - 2 * u[a, c] * v[c, b] / 3
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

u = array([[1, 2], [3, 4]])
v = array([[1, 0], [0, 1]])

{code}

expected = array([[ 0.33333333,  1.66666667], [ 0.0,  1.33333333]])
global diff
diff = linalg.norm(x - expected)

"""
