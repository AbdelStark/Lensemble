"""(eps, delta) DP accountant (RFC-0012 §3; #50).

Pins the budget lifecycle (check-before-release / consume-after-success, fail-closed), validates both
backends against independent closed-form references, the RDP-vs-PRV tightness ordering, and the
infeasible-target failure. The runtime that actually raises PrivacyBudgetExceeded is simulated here
(owned by the participant/coordinator runtime).
"""

from __future__ import annotations

import math

import pytest

from lensemble.errors import ConfigError, LensembleErrorCode, PrivacyBudgetExceeded
from lensemble.privacy import PRVAccountant, RDPAccountant, build_accountant

_ORDERS = range(2, 257)


def _reference_rdp_epsilon(sigma: float, num_rounds: int, delta: float) -> float:
    # Independent full-participation RDP: RDP(a) = T*a/(2 sigma^2); eps = min_a [RDP(a)+ln(1/delta)/(a-1)].
    log_inv = math.log(1.0 / delta)
    return min(
        num_rounds * a / (2.0 * sigma * sigma) + log_inv / (a - 1) for a in _ORDERS
    )


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _reference_gaussian_delta(eps: float, sigma_eff: float) -> float:
    a = 1.0 / (2.0 * sigma_eff)
    return _cdf(-eps * sigma_eff + a) - math.exp(eps) * _cdf(-eps * sigma_eff - a)


# --- backend correctness vs independent references ---


def test_rdp_spent_matches_independent_reference() -> None:
    acc = RDPAccountant()
    sigma, rounds, delta = 1.2, 20, 1e-5
    for _ in range(rounds):
        acc.step(noise_multiplier=sigma)
    assert acc.spent(target_delta=delta) == pytest.approx(
        _reference_rdp_epsilon(sigma, rounds, delta), rel=1e-9
    )


def test_prv_spent_matches_exact_analytic_gaussian() -> None:
    acc = PRVAccountant()
    sigma, rounds, delta = 1.2, 20, 1e-5
    for _ in range(rounds):
        acc.step(noise_multiplier=sigma)
    eps = acc.spent(target_delta=delta)
    sigma_eff = sigma / math.sqrt(rounds)
    # the reported eps reproduces delta(eps) == target_delta on the exact Gaussian curve
    assert _reference_gaussian_delta(eps, sigma_eff) == pytest.approx(delta, rel=1e-6)


def test_prv_is_no_looser_than_rdp_for_same_composition() -> None:
    sigma, rounds, delta = 1.5, 30, 1e-5
    rdp, prv = RDPAccountant(), PRVAccountant()
    for _ in range(rounds):
        rdp.step(noise_multiplier=sigma)
        prv.step(noise_multiplier=sigma)
    assert prv.spent(target_delta=delta) <= rdp.spent(target_delta=delta) + 1e-9


# --- calibration ---


@pytest.mark.parametrize("kind", ["rdp", "prv"])
def test_calibrate_sigma_hits_the_budget_and_is_near_minimal(kind: str) -> None:
    acc = build_accountant(kind)
    target_eps, delta, rounds = 4.0, 1e-5, 50
    sigma = acc.calibrate_sigma(
        target_epsilon=target_eps, target_delta=delta, num_rounds=rounds
    )
    # composing `rounds` steps at the calibrated sigma stays within budget...
    spent_acc = build_accountant(kind)
    for _ in range(rounds):
        spent_acc.step(noise_multiplier=sigma)
    assert spent_acc.spent(target_delta=delta) <= target_eps + 1e-6
    # ...and a slightly smaller sigma would exceed it (near-minimal).
    looser = build_accountant(kind)
    for _ in range(rounds):
        looser.step(noise_multiplier=sigma * 0.9)
    assert looser.spent(target_delta=delta) > target_eps


@pytest.mark.parametrize("kind", ["rdp", "prv"])
def test_calibrate_sigma_infeasible_target_raises(kind: str) -> None:
    acc = build_accountant(kind)
    with pytest.raises(ConfigError):
        acc.calibrate_sigma(target_epsilon=0.0, target_delta=1e-5, num_rounds=10)
    with pytest.raises(ConfigError):
        acc.calibrate_sigma(target_epsilon=1.0, target_delta=2.0, num_rounds=10)
    with pytest.raises(ConfigError):
        acc.calibrate_sigma(target_epsilon=1.0, target_delta=1e-5, num_rounds=0)


# --- budget lifecycle: check-before-release, consume-after-success, fail-closed ---


def _release_round(acc, *, target_eps, delta, sigma, rnd) -> None:
    """Simulate the runtime's release: check budget BEFORE consuming, raise without spending if over."""
    if acc.would_exceed(
        target_epsilon=target_eps, target_delta=delta, noise_multiplier=sigma
    ):
        err = PrivacyBudgetExceeded(
            "planned (eps, delta) budget is spent; refusing the round",
            code=LensembleErrorCode.DP_BUDGET_EXCEEDED,
            remediation="raise epsilon/delta, lower the round count, or accept the stop",
        )
        err.epsilon_spent = acc.spent(target_delta=delta)  # type: ignore[attr-defined]
        err.epsilon_budget = target_eps  # type: ignore[attr-defined]
        err.round = rnd  # type: ignore[attr-defined]
        raise err
    acc.step(noise_multiplier=sigma)  # consume only after the check passes


def test_budget_is_fail_closed_and_stops_without_overspending() -> None:
    acc = RDPAccountant()
    target_eps, delta, sigma = 2.0, 1e-5, 0.8
    released = 0
    with pytest.raises(PrivacyBudgetExceeded) as exc:
        for rnd in range(1000):  # far more than the budget allows
            _release_round(
                acc, target_eps=target_eps, delta=delta, sigma=sigma, rnd=rnd
            )
            released += 1
    # the refused round did not consume budget: spent reflects exactly the released rounds and is <= budget
    assert acc.spent(target_delta=delta) <= target_eps
    assert exc.value.epsilon_spent <= target_eps  # type: ignore[attr-defined]
    assert exc.value.round == released  # type: ignore[attr-defined]


def test_aborted_round_after_check_does_not_consume_budget() -> None:
    acc = RDPAccountant()
    target_eps, delta, sigma = 8.0, 1e-5, 1.0
    # check passes (budget available) but the round aborts before success -> step NOT called
    assert not acc.would_exceed(
        target_epsilon=target_eps, target_delta=delta, noise_multiplier=sigma
    )
    before = acc.spent(target_delta=delta)
    # ...round aborts here; we deliberately do not call acc.step(...)
    assert acc.spent(target_delta=delta) == before == 0.0


# --- the PRV subsampling boundary is explicit (no untightened number) ---


def test_prv_rejects_subsampling() -> None:
    acc = PRVAccountant()
    with pytest.raises(ConfigError):
        acc.step(noise_multiplier=1.0, sample_rate=0.5)


def test_rdp_subsampling_amplifies_privacy() -> None:
    # subsampling (q<1) spends less epsilon than full participation for the same sigma/rounds.
    full, sub = RDPAccountant(), RDPAccountant()
    for _ in range(20):
        full.step(noise_multiplier=1.0, sample_rate=1.0)
        sub.step(noise_multiplier=1.0, sample_rate=0.1)
    assert sub.spent(target_delta=1e-5) < full.spent(target_delta=1e-5)


# --- edge branches: backend selection, PRV guards, zero/loose budget ---


def test_build_accountant_rejects_unknown_backend() -> None:
    with pytest.raises(ConfigError):
        build_accountant("bogus")


def test_prv_step_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ConfigError):
        PRVAccountant().step(noise_multiplier=0.0)


def test_prv_would_exceed_rejects_subsampling() -> None:
    with pytest.raises(ConfigError):
        PRVAccountant().would_exceed(
            target_epsilon=1.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.5
        )


def test_prv_spent_is_zero_before_any_step() -> None:
    assert PRVAccountant().spent(target_delta=1e-5) == 0.0


def test_calibrate_rejects_bad_sample_rate() -> None:
    with pytest.raises(ConfigError):
        RDPAccountant().calibrate_sigma(
            target_epsilon=1.0, target_delta=1e-5, num_rounds=10, sample_rate=1.5
        )


def test_prv_loose_delta_gives_zero_epsilon() -> None:
    # a target delta so loose that delta(0) <= target -> the smallest sufficient epsilon is 0.
    acc = PRVAccountant()
    acc.step(noise_multiplier=1.0)
    assert acc.spent(target_delta=0.999) == 0.0
