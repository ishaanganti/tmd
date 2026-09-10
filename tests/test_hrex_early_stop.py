"""Unit tests for HREX early-stop plumbing.

End-to-end triggering of early stop is left to integration tests (existing HREX tests with
early_stop_tol set).
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tmd.fe.bar import df_from_ukln_by_lambda
from tmd.fe.free_energy import HREXParams, MDParams, compute_total_ns
from tmd.md.hrex import HREXDiagnostics


def _md_params(**overrides):
    kwargs = dict(n_frames=1000, n_eq_steps=10_000, steps_per_frame=400, seed=2026, hrex_params=HREXParams())
    kwargs.update(overrides)
    return MDParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(early_stop_tol=-0.1),
        dict(early_stop_tol=0.0),
        dict(early_stop_tol=0.1, early_stop_check_interval=0),
        dict(n_frames=100, early_stop_tol=0.1, early_stop_min_frames=101),
    ],
)
def test_early_stop_config_validation_rejects_invalid(kwargs):
    with pytest.raises(AssertionError):
        _md_params(**kwargs)


def test_early_stop_disabled_by_default():
    p = _md_params()
    assert p.early_stop_tol is None
    # min_frames validation is intentionally skipped when the feature is off.
    assert _md_params(early_stop_min_frames=999_999).early_stop_min_frames == 999_999


@dataclass
class _FakeInitialState:
    integrator: object


@dataclass
class _FakeIntegrator:
    dt: float = 2.5e-3


@dataclass
class _FakePairBarResult:
    initial_states: list


@dataclass
class _FakeHREXSimulationResult:
    final_result: _FakePairBarResult
    hrex_diagnostics: HREXDiagnostics
    intermediate_results: list


def _fake_result(n_windows, diagnostics):
    states = [_FakeInitialState(_FakeIntegrator()) for _ in range(n_windows)]
    return _FakeHREXSimulationResult(_FakePairBarResult(states), diagnostics, [])


def test_compute_total_ns_uses_recorded_frames_when_present():
    p = _md_params(n_frames=100)
    full = compute_total_ns(_fake_result(3, HREXDiagnostics([], [], n_frames_completed=100)), p)
    half = compute_total_ns(_fake_result(3, HREXDiagnostics([], [], n_frames_completed=50)), p)
    # Fallback path: n_frames_completed=None must behave the same as recording md_params.n_frames.
    fallback = compute_total_ns(_fake_result(3, HREXDiagnostics([], [], n_frames_completed=None)), p)
    assert half < full == fallback


def _synthetic_ukln(n_lambda, n_samples, dg_per_pair_kbt, noise, rng):
    u = np.zeros((n_lambda, 2, 2, n_samples))
    for i in range(n_lambda):
        u[i, 0, 1, :] = dg_per_pair_kbt + rng.normal(0, noise, n_samples)
        u[i, 1, 0, :] = -dg_per_pair_kbt + rng.normal(0, noise, n_samples)
    return u


def test_split_half_bar_distinguishes_converged_from_drifting():
    rng = np.random.default_rng(0)
    # Stationary: both halves should agree well within error and match the true dF (4 * 0.5 = 2.0 kBT).
    stationary = _synthetic_ukln(n_lambda=4, n_samples=2000, dg_per_pair_kbt=0.5, noise=0.05, rng=rng)
    half = stationary.shape[-1] // 2
    df_a, err_a = df_from_ukln_by_lambda(stationary[..., :half])
    df_b, err_b = df_from_ukln_by_lambda(stationary[..., half:])
    assert abs(df_a - df_b) < 5 * max(err_a, err_b)
    assert abs(df_a - 2.0) < 0.1 and abs(df_b - 2.0) < 0.1

    # Drifting: dG shifts partway through, halves must disagree by well more than any reasonable tol.
    first = _synthetic_ukln(n_lambda=4, n_samples=1000, dg_per_pair_kbt=0.0, noise=0.01, rng=rng)
    second = _synthetic_ukln(n_lambda=4, n_samples=1000, dg_per_pair_kbt=1.0, noise=0.01, rng=rng)
    drifting = np.concatenate([first, second], axis=-1)
    half = drifting.shape[-1] // 2
    df_a, _ = df_from_ukln_by_lambda(drifting[..., :half])
    df_b, _ = df_from_ukln_by_lambda(drifting[..., half:])
    assert abs(df_a - df_b) > 1.0
