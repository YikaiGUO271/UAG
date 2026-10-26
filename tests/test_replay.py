import numpy as np
import pytest
from uag import _runtime as api, _core as core
from test_api import nonconvex


def test_real_negative_support_replay_is_paid_and_preserves_live_state():
    meter = api.Meter(nonconvex, 20, 1e-6)
    run = api.Runtime(1., 1e-6, 20, 1, meter)
    run.phase, run.ell = 1, 1.
    outer = api.query(np.array([0.]), run)
    state = core.State(outer)
    journal = core.Journal(0)
    core.prepare(state, outer, .1, False, run, journal)
    entry, entry_ell, before = core.clone(state), run.ell, run.calls
    action, target, support = core.advance(state, outer, .1, run, journal, False)
    assert action == 0 and core.remainder(target, support, .1/4.) < 0.
    forward = run.calls-before
    saved = state.log_mass, state.z.copy(), state.point[0].copy(), run.ell
    u, w = core.replay_pair(entry, entry_ell, target, outer, .1, run, False, forward)
    np.testing.assert_array_equal(u[0], target[0])
    np.testing.assert_array_equal(w[0], support[0])
    assert state.log_mass == saved[0] and run.ell == saved[3]
    np.testing.assert_array_equal(state.z, saved[1])
    np.testing.assert_array_equal(state.point[0], saved[2])
    assert run.calls == meter.calls == run.counts.sum() == 3
    assert run.events[118] == forward == run.counts.reshape(4, 8)[3].sum() == 1


def test_failed_replay_boundary_merges_paid_work(monkeypatch):
    meter = api.Meter(nonconvex, 20, 1e-12)
    run = api.Runtime(1., 1e-12, 20, 1, meter)
    outer = api.query(np.array([0.]), run)
    entry = core.State(outer)

    def faulty_advance(state, outer, lam, shadow, journal, local, replay, target):
        shadow.kind = 3
        point = api.query(np.array([.01]), shadow)
        return 1, point, point

    monkeypatch.setattr(core, "advance", faulty_advance)
    with pytest.raises(ArithmeticError, match="earlier boundary"):
        core.replay_pair(entry, 1., outer, outer, .1, run, False, 2)
    assert run.calls == meter.calls == run.counts.sum() == 2
    assert run.counts.reshape(4, 8)[3].sum() == run.events[118] == 1


def test_replay_arithmetic_exception_merges_paid_work(monkeypatch):
    meter = api.Meter(nonconvex, 20, 1e-12)
    run = api.Runtime(1., 1e-12, 20, 1, meter)
    outer = api.query(np.array([0.]), run)
    entry = core.State(outer)

    def failed_advance(state, outer, lam, shadow, journal, local, replay, target):
        shadow.kind = 3
        api.query(np.array([.01]), shadow)
        raise ArithmeticError("controlled failure after one paid call")

    monkeypatch.setattr(core, "advance", failed_advance)
    with pytest.raises(ArithmeticError, match="after charged work"):
        core.replay_pair(entry, 1., outer, outer, .1, run, False, 2)
    assert run.calls == meter.calls == run.counts.sum() == 2
    assert run.events[118] == 1
