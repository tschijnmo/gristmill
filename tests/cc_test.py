"""Test on computations from CC theories."""

import pytest
from drudge import PartHoleDrudge
from sympy import IndexedBase, Symbol, Rational

from gristmill import optimize, verify_eval_seq, Strategy, get_flop_cost


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

    searched_eval_seq = optimize(targets, substs={p.nv: p.no * 10})

    assert verify_eval_seq(searched_eval_seq, targets)
    assert len(searched_eval_seq) == 2
    searched_cost = get_flop_cost(searched_eval_seq)

    best_eval_seq = optimize(
        targets, substs={p.nv: p.no * 10},
        strategy=Strategy.BEST | Strategy.SUM | Strategy.COMMON
    )
    assert verify_eval_seq(best_eval_seq, targets)
    assert len(best_eval_seq) == 2
    best_cost = get_flop_cost(best_eval_seq)

    assert (best_cost - searched_cost).xreplace({p.no: 1, p.nv: 10}) > 0


def test_ccsd_doubles(parthole_drudge):
    """Test discovery of effective T in CCSD doubles equation.

    The purpose of this test is similar to the CCSD energy test.  Just here the
    more complexity about the external indices necessitates using ``ALL``
    strategy for optimization.
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

    all_eval_seq = optimize(
        targets, substs={p.nv: p.no * 10},
        strategy=Strategy.ALL | Strategy.SUM | Strategy.COMMON
    )

    assert verify_eval_seq(all_eval_seq, targets)
    assert len(all_eval_seq) == 2
    all_cost = get_flop_cost(all_eval_seq)

    best_eval_seq = optimize(
        targets, substs={p.nv: p.no * 10},
        strategy=Strategy.BEST | Strategy.SUM | Strategy.COMMON
    )
    assert verify_eval_seq(best_eval_seq, targets)
    assert len(best_eval_seq) == 2
    best_cost = get_flop_cost(best_eval_seq)

    assert (best_cost - all_cost).xreplace({p.no: 1, p.nv: 10}) > 0


def test_ccsd_doubles_a_terms(parthole_drudge):
    r"""Test optimization of the a tau term in CCSD doubles equation.

    This test is for the optimization of the

    .. math::

        a^{kl}_{ij} \tau^{ab}{kl}

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
        IndexedBase('ai')[k, l, i, j], u[i, c, k, l] * t[c, j]
    )

    a_ = dr.define(
        IndexedBase('a')[k, l, i, j],
        u[k, l, i, j] +
        a_i[k, l, i, j] - a_i[k, l, j, i]
        + u[k, l, c, d] * tau[c, d, i, j]
    )

    tensor = dr.define_einst(
        IndexedBase('r')[a, b, i, j],
        a_[k, l, i, j] * tau[a, b, k, l]
    )
    targets = [tensor]

    # This example should work well both with full backtrack and greedily.
    for drop_cutoff in [-1, 2]:
        eval_seq = optimize(
            targets, substs={p.nv: p.no * 10},
            strategy=Strategy.ALL | Strategy.SUM, drop_cutoff=drop_cutoff
        )
        assert verify_eval_seq(eval_seq, targets)
        # Here we just assert that the final step is a simple product.
        assert len(eval_seq[-1].rhs_terms) == 1
