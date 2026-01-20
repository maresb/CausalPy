# Orthogonal Functions to Zero-Sum Constrained Seasonal Effects

## Overview

This document analyzes the mathematical properties of `ZeroSumNormal` constraints commonly used for seasonal effects in time series models. Using the Riesz representation theorem and Fourier analysis, we derive the closed-form characterization of functions orthogonal to the zero-sum constrained subspace, and examine whether certain time series are "invisible" (unfittable) to such models.

## CausalPy Context: StateSpaceTimeSeries Model

In CausalPy's `StateSpaceTimeSeries` model (from `pymc_models.py`), the `ZeroSumNormal` distribution is used for the `_annual_seasonal` parameter:

```python
_annual_seasonal = pm.ZeroSumNormal(
    "params_freq", sigma=80, dims=annual_dims
)
```

This is part of a state-space formulation that combines:
1. **`LevelTrendComponent`**: Captures the level/mean (DC component) and trend
2. **`FrequencySeasonality`**: Captures periodic variations using Fourier harmonics

The key insight is that **the DC component is handled by the level component, not the seasonal component**. The ZeroSumNormal prior on seasonal Fourier coefficients is an identifiability/regularization constraint, not a limitation on what the model can fit.

## Mathematical Setup

### The Zero-Sum Constraint

Consider a seasonal component $\mathbf{s} = (s_0, s_1, \ldots, s_{n-1})^\top \in \mathbb{R}^n$ representing effects for $n$ periods (e.g., 52 weeks, 12 months). The **zero-sum constraint** requires:

$$\sum_{t=0}^{n-1} s_t = 0$$

This defines a subspace $S \subset \mathbb{R}^n$ of dimension $n-1$:

$$S = \left\{ \mathbf{s} \in \mathbb{R}^n : \mathbf{1}^\top \mathbf{s} = 0 \right\}$$

where $\mathbf{1} = (1, 1, \ldots, 1)^\top$ is the all-ones vector.

### The Orthogonal Complement

Using the standard inner product $\langle \mathbf{f}, \mathbf{g} \rangle = \sum_{t=0}^{n-1} f_t g_t$, we seek the orthogonal complement $S^\perp$—the set of all vectors orthogonal to every vector in $S$.

**Theorem:** The orthogonal complement of the zero-sum subspace is the span of the constant vector:

$$S^\perp = \text{span}\{\mathbf{1}\} = \{c \cdot \mathbf{1} : c \in \mathbb{R}\}$$

**Proof:** 
1. First, note that $\mathbf{1} \perp S$ because for any $\mathbf{s} \in S$:
   $$\langle \mathbf{1}, \mathbf{s} \rangle = \sum_{t=0}^{n-1} 1 \cdot s_t = \sum_{t=0}^{n-1} s_t = 0$$

2. Since $\dim(S) = n-1$ and $\dim(S^\perp) = n - \dim(S) = 1$, we have $S^\perp = \text{span}\{\mathbf{1}\}$. $\square$

## Fourier Analysis Perspective

### Exponential Form

The discrete Fourier transform (DFT) basis vectors are:

$$e_k(t) = \exp\left(\frac{2\pi i k t}{n}\right), \quad k = 0, 1, \ldots, n-1$$

Any periodic signal $\mathbf{f}$ can be written as:

$$f_t = \sum_{k=0}^{n-1} \hat{f}_k \cdot e_k(t) = \sum_{k=0}^{n-1} \hat{f}_k \exp\left(\frac{2\pi i k t}{n}\right)$$

where $\hat{f}_k$ are the Fourier coefficients.

### Zero-Sum Constraint in Fourier Domain

The zero-sum constraint $\sum_{t=0}^{n-1} s_t = 0$ in the Fourier domain becomes:

$$\sum_{t=0}^{n-1} s_t = \sum_{t=0}^{n-1} \sum_{k=0}^{n-1} \hat{s}_k e_k(t) = \sum_{k=0}^{n-1} \hat{s}_k \underbrace{\sum_{t=0}^{n-1} e_k(t)}_{\delta_{k,0} \cdot n}$$

The inner sum equals $n$ when $k=0$ and $0$ otherwise (orthogonality of DFT basis). Thus:

$$\sum_{t=0}^{n-1} s_t = n \cdot \hat{s}_0 = 0 \implies \hat{s}_0 = 0$$

**The zero-sum constraint eliminates the DC (k=0) component!**

### Trigonometric Form

Converting from exponential to trigonometric form using Euler's formula:

$$\exp(i\theta) = \cos(\theta) + i\sin(\theta)$$

A real-valued seasonal signal with zero-sum constraint has the form:

$$s_t = \sum_{k=1}^{\lfloor n/2 \rfloor} \left[ a_k \cos\left(\frac{2\pi k t}{n}\right) + b_k \sin\left(\frac{2\pi k t}{n}\right) \right]$$

Note the sum starts at $k=1$, not $k=0$. The missing $k=0$ term would be the constant $a_0/2$.

### Closed-Form for Orthogonal Functions

**Theorem (via Riesz Representation):** A function $g: \{0, 1, \ldots, n-1\} \to \mathbb{R}$ is orthogonal to all zero-sum constrained functions if and only if $g$ is constant.

**Proof using Riesz Representation:**

In the finite-dimensional Hilbert space $\mathbb{R}^n$ with the standard inner product, every continuous linear functional $\phi: \mathbb{R}^n \to \mathbb{R}$ has a unique representer $\mathbf{g}$ such that:

$$\phi(\mathbf{f}) = \langle \mathbf{g}, \mathbf{f} \rangle \quad \forall \mathbf{f} \in \mathbb{R}^n$$

Consider the linear functional $\phi(\mathbf{s}) = 0$ on the subspace $S$. We want to find $\mathbf{g}$ such that $\langle \mathbf{g}, \mathbf{s} \rangle = 0$ for all $\mathbf{s} \in S$.

The Riesz representer lives in $S^\perp$, which we've shown is $\text{span}\{\mathbf{1}\}$.

Thus, $\mathbf{g} = c \cdot \mathbf{1}$ for some $c \in \mathbb{R}$, meaning $g_t = c$ (constant). $\square$

**In Fourier terms:** The only function with all non-DC harmonics equal to zero is the constant function—the pure DC signal $\hat{g}_k = c \cdot \delta_{k,0}$.

## Implications for Time Series Modeling

### Case 1: Model WITH Intercept (Correct Specification)

Consider the standard seasonal model:

$$y_t = \mu + s_t + \varepsilon_t$$

where:
- $\mu$ is the overall mean (intercept)
- $s_t \sim \text{ZeroSumNormal}$ is the seasonal effect
- $\varepsilon_t$ is noise

In this model:
- The intercept $\mu$ captures the constant (DC) component
- The seasonal $s_t$ captures all other harmonics
- **No time series is "invisible"**—the model spans all of $\mathbb{R}^n$

The zero-sum constraint is purely for **identifiability**: without it, $\mu$ and $s_t$ would be unidentifiable (you could add any constant to $s_t$ and subtract it from $\mu$).

### Case 2: Model WITHOUT Intercept (Problematic)

If someone naively specifies:

$$y_t = s_t + \varepsilon_t$$

with $s_t \sim \text{ZeroSumNormal}$ but **no intercept**, then:

- The model can only capture signals in $S$ (zero-sum signals)
- **A constant offset is "invisible"**—cannot be fit
- Any time series $y_t = c + \tilde{y}_t$ will have the constant $c$ ignored

### The "Invisible" Time Series

The unique (up to scaling) time series that cannot be fit by a pure ZeroSumNormal model is:

$$y_t = c \quad \text{(constant)}$$

Or equivalently, in Fourier terms:
$$\hat{y}_k = c \cdot \delta_{k,0}$$

**This is the function orthogonal to the entire zero-sum constrained subspace.**

### Case 3: StateSpaceTimeSeries (CausalPy Implementation)

The `StateSpaceTimeSeries` model in CausalPy uses a state-space formulation:

```python
# From pymc_models.py
trend = LevelTrendComponent(order=self.level_order)
season = FrequencySeasonality(season_length=self.seasonal_length, name="freq")
combined = trend + season
```

With priors:
```python
_initial_trend = pm.Normal("initial_level_trend", sigma=50, dims=initial_trend_dims)
_annual_seasonal = pm.ZeroSumNormal("params_freq", sigma=80, dims=annual_dims)
```

**Why this is correctly specified:**

1. **`LevelTrendComponent`** captures the level (time-varying mean) including the DC component
2. **`FrequencySeasonality`** captures Fourier harmonics $k \geq 1$ (no DC by design)
3. **`ZeroSumNormal` on Fourier coefficients** constrains: $\sum_i \theta_i = 0$

In this architecture:
- The **DC component is captured by the level component**, not the seasonal
- The seasonal component models **deviations from the level** using Fourier harmonics
- No time series is "invisible" because the level can absorb any constant

**The ZeroSumNormal constraint on `params_freq` is a regularization/identifiability constraint that:**
- Prevents the initial seasonal state from having an arbitrary offset
- This offset would be absorbed by the level component anyway
- Provides a well-defined parameterization for the seasonal Fourier coefficients

## Verification Strategy

To empirically verify this:

1. Create a constant time series $y_t = c$
2. Fit a model with ONLY a ZeroSumNormal seasonal component (no intercept)
3. The model should fail to capture the constant—residuals will equal the constant

Conversely, with a proper intercept:
1. Fit a model with intercept + ZeroSumNormal seasonal
2. The intercept should capture $c$, seasonal effects should be near zero
3. Model fits correctly

## Mathematical Summary

| Constraint | Subspace Dimension | Orthogonal Complement | Missing Component |
|------------|-------------------|----------------------|-------------------|
| $\sum s_t = 0$ | $n-1$ | $\text{span}\{\mathbf{1}\}$ (dim=1) | DC ($k=0$) |

**Key Formulas:**

1. Zero-sum subspace: $S = \ker(\mathbf{1}^\top) = \{\mathbf{s} : \mathbf{1}^\top \mathbf{s} = 0\}$

2. Orthogonal complement: $S^\perp = \text{span}\{\mathbf{1}\}$

3. Projection onto $S$: $\Pi_S(\mathbf{y}) = \mathbf{y} - \bar{y} \cdot \mathbf{1}$ where $\bar{y} = \frac{1}{n}\sum_t y_t$

4. Fourier characterization: $\mathbf{s} \in S \iff \hat{s}_0 = 0$

## Conclusion

**Your premise is mathematically correct, but the CausalPy implementation handles this properly:**

1. **Yes**, there exists a function orthogonal to all zero-sum constrained functions: the constant function.

2. **In CausalPy's `StateSpaceTimeSeries`**, this is handled by the architecture:
   - `LevelTrendComponent` captures the DC/mean component (time-varying level)
   - `FrequencySeasonality` captures oscillatory components (harmonics k ≥ 1)
   - The ZeroSumNormal prior on `params_freq` is applied to Fourier coefficients, not to direct seasonal effects

3. **The constraint IS sensibly implemented:**
   - The level component absorbs any constant/DC signal
   - The seasonal component models periodic deviations from the level
   - ZeroSumNormal prevents identifiability issues in the Fourier coefficient space

4. **No time series is "invisible"** to the `StateSpaceTimeSeries` model because:
   - Constants are captured by the level
   - All periodic variations are captured by the Fourier harmonics
   - The combined model spans the full signal space

**Key insight for state-space models:** Unlike simple regression where you need an explicit intercept term, state-space models with a level component automatically separate the mean from the seasonal variations. The zero-sum constraint on seasonal parameters is a regularization choice, not a limitation.

The zero-sum constraint is mathematically equivalent to saying "seasonal Fourier coefficients should be centered," which is a sensible regularization that works in harmony with the level component.
