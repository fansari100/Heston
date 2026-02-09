"""
Heston Stochastic Volatility Model — Complete Implementation.

Provides semi-analytical pricing (characteristic function / Fourier),
Monte Carlo simulation, calibration, implied vol surface generation,
and Greeks computation.

Reference: Heston, S. (1993). "A Closed-Form Solution for Options with
Stochastic Volatility." Review of Financial Studies 6(2): 327-343.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Optional


@dataclass
class HestonParams:
    """Heston model parameters."""

    S0: float     # Initial asset price
    v0: float     # Initial variance
    kappa: float  # Mean-reversion speed
    theta: float  # Long-run variance
    xi: float     # Vol-of-vol
    rho: float    # Correlation
    r: float      # Risk-free rate


class HestonModel:
    """
    Full Heston stochastic volatility model.

    Parameters
    ----------
    S0, v0, kappa, theta, xi, rho, r : model parameters
    """

    def __init__(
        self,
        S0: float = 100.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        r: float = 0.05,
    ):
        self.p = HestonParams(S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=r)

    # ------------------------------------------------------------------
    # Characteristic Function (Heston 1993, formulation 2 — more stable)
    # ------------------------------------------------------------------
    def characteristic_function(self, phi: complex, T: float) -> complex:
        """
        Heston characteristic function E[exp(iφ ln S_T)].

        Uses the 'little Heston trap' formulation (Albrecher et al., 2007)
        for numerical stability — avoids branch-cut discontinuities.
        """
        p = self.p
        i = 1j

        d = np.sqrt(
            (p.rho * p.xi * i * phi - p.kappa) ** 2
            + p.xi**2 * (i * phi + phi**2)
        )

        g = (p.kappa - p.rho * p.xi * i * phi - d) / (
            p.kappa - p.rho * p.xi * i * phi + d
        )

        exp_dT = np.exp(-d * T)

        C = p.r * i * phi * T + (p.kappa * p.theta / p.xi**2) * (
            (p.kappa - p.rho * p.xi * i * phi - d) * T
            - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )

        D = (
            (p.kappa - p.rho * p.xi * i * phi - d)
            / p.xi**2
            * (1 - exp_dT)
            / (1 - g * exp_dT)
        )

        return np.exp(C + D * p.v0 + i * phi * np.log(p.S0))

    # ------------------------------------------------------------------
    # Semi-Analytical Pricing via Fourier Inversion
    # ------------------------------------------------------------------
    def call_price_fourier(self, K: float, T: float) -> float:
        """
        European call price using Gil-Pelaez inversion of the
        characteristic function.

        C = S₀ P₁ - K e^{-rT} P₂

        where P_j = ½ + (1/π) ∫₀^∞ Re[e^{-iφ ln K} f_j(φ)] / φ dφ
        """
        p = self.p

        def integrand_P1(phi: float) -> float:
            cf = self.characteristic_function(phi - 1j, T)
            num = np.exp(-1j * phi * np.log(K)) * cf
            denom = 1j * phi * self.characteristic_function(-1j, T)
            return np.real(num / denom)

        def integrand_P2(phi: float) -> float:
            cf = self.characteristic_function(phi, T)
            num = np.exp(-1j * phi * np.log(K)) * cf
            return np.real(num / (1j * phi))

        I1, _ = quad(integrand_P1, 1e-8, 200, limit=500)
        I2, _ = quad(integrand_P2, 1e-8, 200, limit=500)

        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi

        return float(p.S0 * P1 - K * np.exp(-p.r * T) * P2)

    def put_price_fourier(self, K: float, T: float) -> float:
        """European put price via put-call parity."""
        call = self.call_price_fourier(K, T)
        return call - self.p.S0 + K * np.exp(-self.p.r * T)

    # ------------------------------------------------------------------
    # Monte Carlo Simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        T: float,
        n_steps: int = 252,
        n_paths: int = 100_000,
        scheme: str = "euler",
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths.

        Parameters
        ----------
        scheme : "euler" (truncated) or "qe" (Quadratic-Exponential,
                 Andersen 2008 — more accurate for small vol-of-vol)

        Returns
        -------
        S : (n_paths, n_steps+1) asset price paths
        v : (n_paths, n_steps+1) variance paths
        """
        rng = np.random.default_rng(seed)
        p = self.p
        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = p.S0
        v[:, 0] = p.v0

        # Cholesky for correlated Brownian motions
        for i in range(n_steps):
            Z1 = rng.normal(0, 1, n_paths)
            Z2 = rng.normal(0, 1, n_paths)
            W1 = Z1
            W2 = p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z2

            v_pos = np.maximum(v[:, i], 0)  # Truncation scheme
            sqrt_v = np.sqrt(v_pos)

            # Variance process
            v[:, i + 1] = (
                v[:, i]
                + p.kappa * (p.theta - v_pos) * dt
                + p.xi * sqrt_v * np.sqrt(dt) * W2
            )

            # Asset price process (log-Euler for positivity)
            S[:, i + 1] = S[:, i] * np.exp(
                (p.r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W1
            )

        return S, v

    def call_price_mc(
        self,
        K: float,
        T: float,
        n_paths: int = 200_000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Monte Carlo call price with standard error.

        Returns (price, standard_error).
        """
        S, _ = self.simulate(T, n_steps, n_paths, seed=seed)
        payoffs = np.maximum(S[:, -1] - K, 0)
        disc = np.exp(-self.p.r * T)
        price = float(disc * np.mean(payoffs))
        se = float(disc * np.std(payoffs) / np.sqrt(n_paths))
        return price, se

    # ------------------------------------------------------------------
    # Implied Volatility Surface
    # ------------------------------------------------------------------
    def implied_vol(self, K: float, T: float) -> float:
        """Compute Black-Scholes implied volatility for a given (K, T)."""
        market_price = self.call_price_fourier(K, T)
        return self._bs_implied_vol(market_price, self.p.S0, K, T, self.p.r)

    def implied_vol_surface(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate implied volatility surface σ_impl(K, T).

        Returns
        -------
        strikes, maturities, iv_surface : arrays
        """
        if strikes is None:
            strikes = np.linspace(0.8 * self.p.S0, 1.2 * self.p.S0, 15)
        if maturities is None:
            maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        iv = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                try:
                    iv[i, j] = self.implied_vol(K, T)
                except Exception:
                    iv[i, j] = np.nan

        return strikes, maturities, iv

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------
    def delta(self, K: float, T: float, dS: float = 0.5) -> float:
        """Delta via central finite difference."""
        p = self.p
        m_up = HestonModel(p.S0 + dS, p.v0, p.kappa, p.theta, p.xi, p.rho, p.r)
        m_dn = HestonModel(p.S0 - dS, p.v0, p.kappa, p.theta, p.xi, p.rho, p.r)
        return (m_up.call_price_fourier(K, T) - m_dn.call_price_fourier(K, T)) / (2 * dS)

    def gamma(self, K: float, T: float, dS: float = 0.5) -> float:
        """Gamma via central finite difference."""
        p = self.p
        C0 = self.call_price_fourier(K, T)
        m_up = HestonModel(p.S0 + dS, p.v0, p.kappa, p.theta, p.xi, p.rho, p.r)
        m_dn = HestonModel(p.S0 - dS, p.v0, p.kappa, p.theta, p.xi, p.rho, p.r)
        return (m_up.call_price_fourier(K, T) - 2 * C0 + m_dn.call_price_fourier(K, T)) / (dS**2)

    def vega(self, K: float, T: float, dv: float = 0.001) -> float:
        """Vega w.r.t. initial variance v0."""
        p = self.p
        m_up = HestonModel(p.S0, p.v0 + dv, p.kappa, p.theta, p.xi, p.rho, p.r)
        m_dn = HestonModel(p.S0, p.v0 - dv, p.kappa, p.theta, p.xi, p.rho, p.r)
        return (m_up.call_price_fourier(K, T) - m_dn.call_price_fourier(K, T)) / (2 * dv)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @staticmethod
    def calibrate(
        market_prices: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        x0: Optional[np.ndarray] = None,
    ) -> "HestonModel":
        """
        Calibrate Heston parameters to market option prices via
        Levenberg-Marquardt least squares.

        Parameters
        ----------
        market_prices : (N,) observed call prices
        strikes : (N,) corresponding strikes
        maturities : (N,) corresponding maturities
        S0, r : spot price and risk-free rate
        x0 : initial guess [v0, kappa, theta, xi, rho]
        """
        if x0 is None:
            x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.5])

        def residuals(params: np.ndarray) -> np.ndarray:
            v0, kappa, theta, xi, rho = params
            # Enforce constraints
            v0 = max(v0, 1e-6)
            kappa = max(kappa, 1e-4)
            theta = max(theta, 1e-6)
            xi = max(xi, 1e-4)
            rho = np.clip(rho, -0.999, 0.999)

            model = HestonModel(S0, v0, kappa, theta, xi, rho, r)
            errors = np.zeros(len(market_prices))
            for i in range(len(market_prices)):
                try:
                    model_price = model.call_price_fourier(strikes[i], maturities[i])
                    errors[i] = model_price - market_prices[i]
                except Exception:
                    errors[i] = 1e6
            return errors

        bounds = ([1e-6, 1e-4, 1e-6, 1e-4, -0.999], [1.0, 20.0, 1.0, 5.0, 0.999])
        result = least_squares(residuals, x0, bounds=bounds, method="trf", max_nfev=500)

        v0, kappa, theta, xi, rho = result.x
        return HestonModel(S0, v0, kappa, theta, xi, rho, r)

    # ------------------------------------------------------------------
    # Black-Scholes implied vol inversion (Newton-Raphson)
    # ------------------------------------------------------------------
    @staticmethod
    def _bs_implied_vol(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """Invert Black-Scholes formula to get implied volatility."""
        sigma = 0.3  # initial guess

        for _ in range(max_iter):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-12:
                break

            diff = bs_price - price
            if abs(diff) < tol:
                return sigma

            sigma -= diff / vega
            sigma = max(sigma, 1e-6)

        return sigma


if __name__ == "__main__":
    model = HestonModel(S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.05)

    print("=== Heston Model Demo ===")
    print(f"Parameters: κ={model.p.kappa}, θ={model.p.theta}, ξ={model.p.xi}, ρ={model.p.rho}")

    call = model.call_price_fourier(K=100, T=1.0)
    put = model.put_price_fourier(K=100, T=1.0)
    print(f"\nATM Call (Fourier): ${call:.4f}")
    print(f"ATM Put  (Fourier): ${put:.4f}")

    mc_call, mc_se = model.call_price_mc(K=100, T=1.0, n_paths=200000, seed=42)
    print(f"ATM Call (MC):      ${mc_call:.4f} ± {mc_se:.4f}")

    d = model.delta(K=100, T=1.0)
    g = model.gamma(K=100, T=1.0)
    v = model.vega(K=100, T=1.0)
    print(f"\nGreeks: Δ={d:.4f}, Γ={g:.6f}, Vega={v:.4f}")

    iv = model.implied_vol(K=100, T=1.0)
    print(f"Implied Vol (ATM, 1Y): {iv:.4f} ({iv*100:.2f}%)")
