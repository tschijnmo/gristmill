"""Tests about optimizations of problems from coupled-cluster theories.
"""

import pytest
from drudge import PartHoleDrudge
from sympy import IndexedBase, Symbol, Rational

from gristmill import optimize, verify_eval_seq, ContrStrat, get_flop_cost


@pytest.fixture(scope='module')
def parthole_drudge(spark_ctx):
    """The particle-hold drudge."""
    dr = PartHoleDrudge(spark_ctx)
    return dr


def test_ccd_doubles_terms(parthole_drudge):
    """Test optimization of selected terms in CCD equations.

    This purpose of this test is mostly on the treatment of term in the present
    of symmetries.
    """

    dr = parthole_drudge
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

    assert verify_eval_seq(eval_seq, targets)


def test_ccsd_singles_terms(parthole_drudge):
    """Test selected terms in CCSD singles equation.

    The purpose of this test is the capability of recognition of repeated
    appearance of the same summation intermediates.
    """

    dr = parthole_drudge
    p = dr.names

    a, b, c = p.V_dumms[:3]
    i, j, k = p.O_dumms[:3]
    u = dr.two_body
    f = dr.fock
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    r = IndexedBase('r')
    tensor = dr.define_einst(
        r[a, i],
        t[a, b, i, j] * u[j, k, b, c] * t[c, k] + t[a, b, i, j] * f[j, b]
        - t[a, j] * t[b, i] * f[j, b]
        - t[a, j] * t[b, i] * t[c, k] * u[j, k, b, c]
    )
    targets = [tensor]

    eval_seq = optimize(targets, substs={p.nv: p.no * 10})

    assert verify_eval_seq(eval_seq, targets)
    assert len(eval_seq) == 4


def test_ccsd_energy(parthole_drudge):
    """Test discovery of effective T in CCSD energy equation.

    The purpose of this test is the capability of using locally non-optimal
    contractions in the final summation optimization.  The equation is not CCSD
    energy equation exactly.
    """

    dr = parthole_drudge
    p = dr.names

    a, b = p.V_dumms[:2]
    i, j = p.O_dumms[:2]
    u = dr.two_body
    t = IndexedBase('t')

    energy = dr.define_einst(
        Symbol('e'),
        u[i, j, a, b] * t[a, b, i, j] * Rational(1, 2)
        + u[i, j, a, b] * t[a, i] * t[b, j]
    )
    targets = [energy]

    trav_eval_seq = optimize(targets, substs={p.nv: p.no * 10})

    assert verify_eval_seq(trav_eval_seq, targets)
    assert len(trav_eval_seq) == 2
    trav_cost = get_flop_cost(trav_eval_seq)

    opt_eval_seq = optimize(
        targets, substs={p.nv: p.no * 10}, contr_strat=ContrStrat.OPT
    )
    assert verify_eval_seq(opt_eval_seq, targets)
    assert len(opt_eval_seq) == 2
    opt_cost = get_flop_cost(opt_eval_seq)

    assert (opt_cost - trav_cost).xreplace({p.no: 1, p.nv: 10}) > 0


def test_ccsd_doubles(parthole_drudge):
    """Test discovery of effective T in CCSD doubles equation.

    The purpose of this test is similar to the CCSD energy test.  Just here the
    more complexity about the external indices necessitates using ``EXHAUST``
    strategy for optimization.  Also the usage of fully-given numeric size is
    also tested here.
    """

    dr = parthole_drudge
    p = dr.names

    a, b, c, d = p.V_dumms[:4]
    i, j = p.O_dumms[:2]
    u = dr.two_body
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    tensor = dr.define_einst(
        IndexedBase('r')[a, b, i, j],
        t[c, d, i, j] * u[a, b, c, d] + u[a, b, c, d] * t[c, i] * t[d, j]
    )
    targets = [tensor]

    exhaust_eval_seq = optimize(
        targets, substs={p.nv: p.no * 10}, contr_strat=ContrStrat.EXHAUST
    )

    assert verify_eval_seq(exhaust_eval_seq, targets)
    assert len(exhaust_eval_seq) == 2
    exhaust_cost = get_flop_cost(exhaust_eval_seq)

    opt_eval_seq = optimize(targets, substs={p.nv: p.no * 10})
    assert verify_eval_seq(opt_eval_seq, targets)
    assert len(opt_eval_seq) == 2
    opt_cost = get_flop_cost(opt_eval_seq)

    assert (opt_cost - exhaust_cost).xreplace({p.no: 1, p.nv: 10}) > 0


def test_ccsd_doubles_complex_terms(parthole_drudge):
    r"""Test optimization of the a/b tau term in CCSD doubles equation.

    This test is for the optimization of the

    .. math::

        a^{kl}_{ij} \tau^{ab}{kl} + b^{ab}_{cd} \tau^{cd}_{ij}

    in Equation (8) of GE Scuseria and HF Schaefer: J Chem Phys 90 (7) 1989.

    """

    dr = parthole_drudge
    p = dr.names

    a, b, c, d = p.V_dumms[:4]
    i, j, k, l = p.O_dumms[:4]
    u = dr.two_body
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    tau = dr.define_einst(
        IndexedBase('tau')[a, b, i, j],
        Rational(1, 2) * t[a, b, i, j] + t[a, i] * t[b, j]
    )

    a_i = dr.define_einst(
        IndexedBase('AI')[k, l, i, j], u[i, c, k, l] * t[c, j]
    )

    a_ = dr.define(
        IndexedBase('A')[k, l, i, j],
        u[k, l, i, j] +
        a_i[k, l, i, j] - a_i[k, l, j, i]
        + u[k, l, c, d] * tau[c, d, i, j]
    )

    a_term = dr.einst(a_[k, l, i, j] * tau[a, b, k, l])

    b_i = dr.define_einst(
        IndexedBase('BI')[a, b, c, d], u[a, k, c, d] * t[b, k]
    )

    b_ = dr.define_einst(
        IndexedBase('B')[a, b, c, d],
        u[a, b, c, d] - b_i[a, b, c, d] + b_i[b, a, c, d]
    )

    b_term = dr.einst(b_[a, b, c, d] * tau[c, d, i, j])

    substs = {
        p.no: 1000,
        p.nv: 1100
    }

    # Simple a term or b term should work well both with full backtrack and
    # greedily.
    for term in [a_term, b_term]:
        tensor = dr.define_einst(IndexedBase('r')[a, b, i, j], term)
        targets = [tensor]
        for drop_cutoff in [-1, 2]:
            eval_seq = optimize(
                targets, substs=substs,
                contr_strat=ContrStrat.EXHAUST, drop_cutoff=drop_cutoff
            )
            assert verify_eval_seq(eval_seq, targets)
            # Here we just assert that the final step is a simple product.
            assert len(eval_seq[-1].rhs_terms) == 1

    for drop_cutoff in [-1, 2]:
        tensor = dr.define_einst(IndexedBase('r')[a, b, i, j], a_term + b_term)
        targets = [tensor]
        eval_seq = optimize(
            targets, substs={p.nv: p.no * 1.1},
            contr_strat=ContrStrat.EXHAUST, drop_cutoff=drop_cutoff
        )
        assert verify_eval_seq(eval_seq, targets)
        # Here we assert that the two products are separately factored.
        assert len(eval_seq[-1].rhs_terms) == 2


def test_ccsd_pij_term(parthole_drudge):
    """Test Pij term in the CCSD doubles equation.

    Currently here we only test the correctness.
    """

    dr = parthole_drudge
    p = dr.names

    a, b, c, d = p.V_dumms[:4]
    i, j, k, l = p.O_dumms[:4]
    u = dr.two_body
    t = IndexedBase('t')
    dr.set_dbbar_base(t, 2)

    r = IndexedBase('r')
    f = IndexedBase('f')

    targets = [dr.define_einst(
        r[a, b, i, j],
        - 2 * t[c, l] * t[b, a, k, i] * u[k, l, c, j]
        + 2 * f[k, i] * t[a, b, k, j]
        - 2 * t[c, i] * u[a, b, c, j]
        + t[b, a, k, i] * t[c, d, j, l] * u[k, l, c, d]
        - 2 * t[c, l] * t[d, j] * t[b, a, k, i] * u[k, l, c, d]
        + 2 * f[k, c] * t[c, j] * t[b, a, k, i]
    )]

    drop_cutoff = -1
    eval_seq = optimize(
        targets, substs={p.nv: p.no * 1.1},
        contr_strat=ContrStrat.EXHAUST, drop_cutoff=drop_cutoff
    )
    verify_eval_seq(eval_seq, targets)
