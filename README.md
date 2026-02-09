# Heston Stochastic Volatility Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Implementation of the Heston (1993) stochastic volatility model for option pricing, featuring Monte Carlo simulation, semi-analytical Fourier methods (characteristic function), and calibration to market implied volatility surfaces.**

## Model

```
dS_t = r S_t dt + √v_t S_t dW_1
dv_t = κ(θ - v_t) dt + ξ √v_t dW_2
dW_1 · dW_2 = ρ dt
```

| Parameter | Description |
|-----------|-------------|
| `S_0` | Initial asset price |
| `v_0` | Initial variance |
| `κ` (kappa) | Mean-reversion speed of variance |
| `θ` (theta) | Long-run variance level |
| `ξ` (xi) | Volatility of variance (vol-of-vol) |
| `ρ` (rho) | Correlation between asset and variance Brownian motions |
| `r` | Risk-free rate |

## Features

- **Characteristic function** pricing via Fourier inversion (Carr-Madan FFT)
- **Monte Carlo** simulation with full/truncated Euler and QE discretization
- **Calibration** to market implied volatilities using Levenberg-Marquardt
- **Implied volatility surface** generation (strike × maturity grid)
- **Greeks** computation (Delta, Gamma, Vega, Volga, Vanna)

## Quick Start

```python
from heston import HestonModel, HestonCalibrator

model = HestonModel(S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, r=0.05)

# Semi-analytical price via characteristic function
call_price = model.call_price_fourier(K=105, T=1.0)
print(f"Call: ${call_price:.4f}")

# Monte Carlo price
mc_price, mc_se = model.call_price_mc(K=105, T=1.0, n_paths=200000)
print(f"MC Call: ${mc_price:.4f} ± {mc_se:.4f}")

# Implied volatility surface
strikes, maturities, iv_surface = model.implied_vol_surface()
```

## Author

**Ricky Ansari** — [GitHub](https://github.com/fansari100) | [Website](https://rickyansari.com)
