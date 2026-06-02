"""lensemble.privacy.accountant — the ``(eps, delta)`` differential-privacy accountant (RFC-0012 §3).

Composes the per-round Gaussian mechanism over the **planned** round count, calibrates the noise
multiplier ``sigma`` to a target budget, and reports the cumulative ``epsilon`` spent. The accountant is
the bookkeeping half of DP; the clip+noise mechanism (``INV-DP-BOUND``) is owned by
``lensemble.privacy.dp``.

Two self-contained reference backends satisfy the :class:`Accountant` protocol (no opacus dependency — it
is an optional, host-owned extra):

- :class:`RDPAccountant` — Rényi-DP, the development backend. Accumulates the sampled-Gaussian RDP at a
  grid of integer orders (Mironov 2019, "Rényi DP of the Sampled Gaussian Mechanism"), then converts to
  ``(eps, delta)`` by ``eps = min_alpha [ RDP(alpha) + ln(1/delta)/(alpha-1) ]``. The RDP curve is an
  **upper** bound, so it never under-reports privacy loss.
- :class:`PRVAccountant` — the tighter reporting backend. For full participation it composes the Gaussian
  mechanism **exactly** via the analytic Gaussian mechanism (Balle & Wang 2018): ``T`` rounds at
  multiplier ``sigma`` compose to a single Gaussian with effective multiplier ``sigma/sqrt(T)``
  (heterogeneous ``sigma`` via ``1/sigma_eff^2 = sum_i 1/sigma_i^2``). Being exact, it reports ``eps`` no
  larger than RDP for the same composition (the tightness ordering). Poisson subsampling
  (``sample_rate < 1``) under an exact privacy-loss distribution is a deferred Open Question; this backend
  requires ``sample_rate == 1.0`` and raises ``ConfigError`` otherwise rather than report an untightened
  number.

Lifecycle (fail-closed): a caller checks :meth:`would_exceed` **before** releasing a round (and raises
``PrivacyBudgetExceeded`` without consuming budget) and calls :meth:`step` **only after** a successful
round, so a refused or aborted round never spends budget and the released ``(eps, delta)`` is never
exceeded. The active backend is recorded in the ``RunManifest``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from lensemble.errors import ConfigError, LensembleErrorCode

# Integer RDP orders. A wide grid so the eps = min_alpha conversion is tight for small delta.
_ORDERS: tuple[int, ...] = tuple(range(2, 257))


@runtime_checkable
class Accountant(Protocol):
    """The ``(eps, delta)`` accountant contract (RFC-0012 §3). Stateful over a sequence of rounds."""

    def calibrate_sigma(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        num_rounds: int,
        sample_rate: float = 1.0,
    ) -> float:
        """Smallest ``noise_multiplier`` whose composition over ``num_rounds`` stays within the budget."""
        ...

    def step(self, *, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        """Account for one successful round's release at ``noise_multiplier`` and ``sample_rate``."""
        ...

    def spent(self, *, target_delta: float) -> float:
        """Cumulative ``epsilon`` spent so far at the fixed ``target_delta``."""
        ...

    def would_exceed(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> bool:
        """``True`` iff one more round at this ``sigma`` pushes spent ``epsilon`` past ``target_epsilon``."""
        ...


def _validate_budget(epsilon: float, delta: float, num_rounds: int, q: float) -> None:
    """Reject an infeasible/invalid budget target with :class:`ConfigError` (``CONFIG_INVALID``)."""
    bad = None
    if not epsilon > 0.0:
        bad = f"target_epsilon must be > 0, got {epsilon}"
    elif not 0.0 < delta < 1.0:
        bad = f"target_delta must be in (0, 1), got {delta}"
    elif num_rounds < 1:
        bad = f"num_rounds must be >= 1, got {num_rounds}"
    elif not 0.0 < q <= 1.0:
        bad = f"sample_rate must be in (0, 1], got {q}"
    if bad is not None:
        raise ConfigError(
            bad,
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="set a positive epsilon, a delta in (0,1), >=1 rounds, and a sample_rate in (0,1]",
        )


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def _sampled_gaussian_rdp(order: int, sigma: float, q: float) -> float:
    """One step's sampled-Gaussian RDP at an integer ``order`` (Mironov 2019). q=1 -> ``order/(2 sigma^2)``."""
    if sigma <= 0.0:
        return math.inf
    if q >= 1.0:
        return order / (2.0 * sigma * sigma)
    # log A_order = logsumexp_k [ log C(order,k) + (order-k) log(1-q) + k log q + k(k-1)/(2 sigma^2) ]
    log_terms = []
    for k in range(order + 1):
        log_coeff = (
            math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1)
        )
        term = (
            log_coeff
            + (order - k) * math.log1p(-q)
            + k * math.log(q)
            + (k * (k - 1)) / (2.0 * sigma * sigma)
        )
        log_terms.append(term)
    return _logsumexp(log_terms) / (order - 1)


def _rdp_to_epsilon(rdp_by_order: dict[int, float], delta: float) -> float:
    """Convert accumulated RDP to ``eps`` at ``delta``: ``min_alpha [RDP(alpha) + ln(1/delta)/(alpha-1)]``.

    With no composition (all RDP zero) the privacy loss is exactly zero — the ``ln(1/delta)/(alpha-1)``
    conversion term is a finite-grid artifact that vanishes as ``alpha -> inf``, so it is dropped here.
    """
    if max(rdp_by_order.values()) == 0.0:
        return 0.0
    log_inv_delta = math.log(1.0 / delta)
    return min(rdp + log_inv_delta / (order - 1) for order, rdp in rdp_by_order.items())


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _analytic_gaussian_delta(epsilon: float, sigma_eff: float) -> float:
    """Exact ``delta(epsilon)`` of the Gaussian mechanism at sensitivity 1, noise ``sigma_eff`` (Balle-Wang)."""
    if sigma_eff == math.inf:
        return 0.0
    a = 1.0 / (2.0 * sigma_eff)
    return _std_normal_cdf(-epsilon * sigma_eff + a) - math.exp(
        epsilon
    ) * _std_normal_cdf(-epsilon * sigma_eff - a)


def _invert_analytic_gaussian(delta: float, sigma_eff: float) -> float:
    """Smallest ``epsilon`` with ``delta(epsilon) <= delta`` for the Gaussian mechanism (monotone bisect)."""
    if sigma_eff == math.inf:
        return 0.0
    if _analytic_gaussian_delta(0.0, sigma_eff) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while _analytic_gaussian_delta(hi, sigma_eff) > delta:
        hi *= 2.0
        if hi > 1e9:
            return hi
    for _ in range(100):  # bisection to ~1e-9 relative
        mid = 0.5 * (lo + hi)
        if _analytic_gaussian_delta(mid, sigma_eff) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def _bisect_sigma(epsilon_of_sigma, target_epsilon: float) -> float:
    """Smallest ``sigma`` with ``epsilon_of_sigma(sigma) <= target_epsilon`` (eps decreasing in sigma)."""
    lo, hi = 1e-3, 1.0
    while epsilon_of_sigma(hi) > target_epsilon:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if epsilon_of_sigma(mid) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


class RDPAccountant:
    """Rényi-DP accountant over a grid of integer orders (the development backend; RFC-0012 §3).

    Lifecycle: :meth:`would_exceed` before release, :meth:`step` only after a successful round. The RDP
    curve is an upper bound, so reported ``eps`` is conservative (never under-reported).
    """

    def __init__(self) -> None:
        self._rdp: dict[int, float] = dict.fromkeys(_ORDERS, 0.0)

    def step(self, *, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        for order in _ORDERS:
            self._rdp[order] += _sampled_gaussian_rdp(
                order, noise_multiplier, sample_rate
            )

    def spent(self, *, target_delta: float) -> float:
        return _rdp_to_epsilon(self._rdp, target_delta)

    def would_exceed(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> bool:
        hypothetical = {
            order: self._rdp[order]
            + _sampled_gaussian_rdp(order, noise_multiplier, sample_rate)
            for order in _ORDERS
        }
        return _rdp_to_epsilon(hypothetical, target_delta) > target_epsilon

    def calibrate_sigma(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        num_rounds: int,
        sample_rate: float = 1.0,
    ) -> float:
        _validate_budget(target_epsilon, target_delta, num_rounds, sample_rate)

        def eps_of(sigma: float) -> float:
            rdp = {
                order: num_rounds * _sampled_gaussian_rdp(order, sigma, sample_rate)
                for order in _ORDERS
            }
            return _rdp_to_epsilon(rdp, target_delta)

        return _bisect_sigma(eps_of, target_epsilon)


class PRVAccountant:
    """Privacy-loss-distribution accountant — exact analytic Gaussian for full participation (RFC-0012 §3).

    Reports ``eps`` no larger than :class:`RDPAccountant` for the same composition (RDP is an upper bound
    on this exact value). Requires ``sample_rate == 1.0``; the exact subsampled PLD is a deferred Open
    Question and raises ``ConfigError`` rather than report an untightened number.
    """

    def __init__(self) -> None:
        self._inv_var_sum = 0.0  # sum_i 1/sigma_i^2

    @staticmethod
    def _require_full(sample_rate: float) -> None:
        if sample_rate != 1.0:
            raise ConfigError(
                f"PRVAccountant requires sample_rate == 1.0 (got {sample_rate}); the exact subsampled "
                "privacy-loss-distribution backend is a deferred RFC-0012 Open Question",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="use RDPAccountant for subsampled amplification, or release at full participation",
            )

    def _epsilon(self, inv_var_sum: float, target_delta: float) -> float:
        if inv_var_sum <= 0.0:
            return 0.0
        sigma_eff = 1.0 / math.sqrt(inv_var_sum)
        return _invert_analytic_gaussian(target_delta, sigma_eff)

    def step(self, *, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        self._require_full(sample_rate)
        if noise_multiplier <= 0.0:
            raise ConfigError(
                f"noise_multiplier must be > 0, got {noise_multiplier}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="release with a positive Gaussian noise multiplier",
            )
        self._inv_var_sum += 1.0 / (noise_multiplier * noise_multiplier)

    def spent(self, *, target_delta: float) -> float:
        return self._epsilon(self._inv_var_sum, target_delta)

    def would_exceed(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> bool:
        self._require_full(sample_rate)
        hypothetical = self._inv_var_sum + 1.0 / (noise_multiplier * noise_multiplier)
        return self._epsilon(hypothetical, target_delta) > target_epsilon

    def calibrate_sigma(
        self,
        *,
        target_epsilon: float,
        target_delta: float,
        num_rounds: int,
        sample_rate: float = 1.0,
    ) -> float:
        _validate_budget(target_epsilon, target_delta, num_rounds, sample_rate)
        self._require_full(sample_rate)

        def eps_of(sigma: float) -> float:
            return self._epsilon(num_rounds / (sigma * sigma), target_delta)

        return _bisect_sigma(eps_of, target_epsilon)


def build_accountant(kind: str) -> Accountant:
    """Construct the config-selected accountant backend (``"rdp"`` | ``"prv"``; RFC-0012 §5)."""
    if kind == "rdp":
        return RDPAccountant()
    if kind == "prv":
        return PRVAccountant()
    raise ConfigError(
        f"unknown accountant backend {kind!r}; expected 'rdp' or 'prv'",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation="set privacy.accountant to 'rdp' or 'prv'",
    )
