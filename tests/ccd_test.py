"""Test on CCD equations."""

from drudge import PartHoleDrudge
from sympy import IndexedBase

from gristmill import optimize, verify_eval_seq


def test_ccd_doubles_terms(spark_ctx):
    """Test optimization of selected terms in CCD equations.

    This purpose of this test is mostly on the treatment of term in the present
    of symmetries.
    """

    dr = PartHoleDrudge(spark_ctx)
    p = dr.names

    a, b, c, d = p.V_dumms[:4]
    i, j, k, l = p.O_dumms[:4]
    u = dr.two_body
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    r = IndexedBase('r')
    tensor = dr.define_einst(
        r[a, b, i, j],
        + t[a, b, l, j] * t[c, d, i, k] * u[k, l, c, d]
        + t[a, d, i, j] * t[b, c, k, l] * u[k, l, c, d]
        - t[a, b, i, l] * t[c, d, k, j] * u[k, l, c, d]
        - t[a, c, k, l] * t[b, d, i, j] * u[k, l, c, d]
    )
    targets = [tensor]

    eval_seq = optimize(targets, substs={p.nv: p.no * 10})

    verify_eval_seq(eval_seq, targets)
