"""
Benchmark of automatic code for a CCD mosaic term.

This example shows a trivial automatic optimization and code generation of the
evaluation of a mosaic term in CCD doubles amplitude equations.  Performance of
different printers are benchmarked.

"""

import subprocess
import sys
import types

from drudge import PartHoleDrudge
from jinja2 import Template
from pyspark import SparkConf, SparkContext
from sympy import IndexedBase

from gristmill import optimize, get_flop_cost, FortranPrinter, EinsumPrinter

#
# Set up the problem.
#

NVNO_RATIO = 1


def get_eval_seq():
    """Get the evaluation sequence for the benchmark."""
    conf = SparkConf().setAppName('CCD-mosaic')
    ctx = SparkContext(conf=conf)

    dr = PartHoleDrudge(ctx)
    dr.full_simplify = False
    p = dr.names

    a, b, c, d = p.V_dumms[:4]
    i, j, k, l = p.O_dumms[:4]
    u = dr.two_body
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    r = IndexedBase('r')
    tensor = dr.define_einst(
        r[a, b, i, j],
        t[a, b, l, j] * t[c, d, i, k] * u[k, l, c, d]
    )
    targets = [tensor]
    eval_seq = optimize(
        targets, substs={p.nv: p.no * NVNO_RATIO}, interm_fmt='tau{}'
    )

    return eval_seq, types.SimpleNamespace(no=p.no, nv=p.nv)


def main():
    """The main driver.

    The commandline arguments will be interpreted as the NO sizes to be tested,
    the timing and FLOPS result will be dumped into files ``timing`` and
    ``flops``.
    """

    nos = [int(i) for i in sys.argv[1:]]
    timings = []
    flops = []

    eval_seq, symbs = get_eval_seq()
    flop_cost_expr = get_flop_cost(eval_seq)

    for no in nos:
        print('Benchmark NO={}'.format(no))
        nv = no * NVNO_RATIO
        flop_cost = flop_cost_expr.xreplace({symbs.no: no, symbs.nv: nv})

        new_timings = tuple(
            run_job(eval_seq, no, nv)
            for run_job in [run_fortran, run_einsum]
        )
        timings.append(new_timings)
        flops.append(tuple(
            flop_cost / i for i in new_timings
        ))
        continue

    with open('timing', 'w') as fp:
        for i, j in zip(nos, timings):
            print(i, *j, file=fp)

    with open('flops', 'w') as fp:
        for i, j in zip(nos, flops):
            print(i, *j, file=fp)
            continue


def run_fortran(eval_seq, no, nv):
    """Run the Fortran job."""

    printer = FortranPrinter()
    decls, evals = printer.print_decl_eval(eval_seq)
    fortran_code = _FORTRAN_TEMPLATE.render(
        no=no, nv=nv, decls=decls, evals=evals
    )

    with open('ccdmos_run.f90', 'w') as fp:
        fp.write(fortran_code)

    stat = subprocess.run([
        'gfortran', '-fopenmp', '-o', 'f',
        '-Wl,-stack_size', '-Wl,1000000000',
        'ccdmos_run.f90'
    ])
    assert stat.returncode == 0

    stat = subprocess.run(['./f'], stdout=subprocess.PIPE)
    timing = float(stat.stdout.decode())

    return timing


_FORTRAN_TEMPLATE = Template("""
program main
implicit none

integer, parameter :: no = {{ no }}
integer, parameter :: nv = {{ nv }}
integer :: i, j, k, l, a, b, c, d

real, dimension(no, no, nv, nv) :: u
real, dimension(nv, nv, no, no) :: t

real :: beg_time, end_time

{% for decl in decls %}{{ decl }}
{% endfor %}

call random_number(u)
call random_number(t)

call cpu_time(beg_time)

{% for eval_ in evals %}{{ eval_ }}
{% endfor %}

call cpu_time(end_time)

write(*, *) end_time - beg_time

end program main
""")


def run_einsum(eval_seq, no, nv):
    """Run NumPy einsum job."""

    printer = EinsumPrinter()
    evals = printer.print_eval(eval_seq, base_indent=0)

    code = _EINSUM_TEMPLATE.render(no=no, nv=nv, evals=evals)
    with open('ccdmos_run.py', 'w') as fp:
        fp.write(code)

    stat = subprocess.run(
        ['python3', './ccdmos_run.py'], stdout=subprocess.PIPE
    )
    timing = float(stat.stdout.decode())

    return timing


_EINSUM_TEMPLATE = Template("""
import time

from numpy import einsum
from numpy import random

no = {{ no }}
nv = {{ nv }}

u = random.rand(no, no, nv, nv)
t = random.rand(nv, nv, no, no)

beg_time = time.time()

{{ evals }}

print(time.time() - beg_time)

""")

if __name__ == '__main__':
    main()
