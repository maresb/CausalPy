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

**⚠️ IMPORTANT CORRECTION:** The analysis below identifies that the `ZeroSumNormal` constraint on Fourier coefficients is **likely a bug**, not a sensible identifiability constraint. See the "Critical Analysis" section for details.

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

### Case 3: StateSpaceTimeSeries (CausalPy Implementation) — CRITICAL ANALYSIS

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

#### The Fourier Basis

The `FrequencySeasonality` component uses a Fourier representation:

$$\gamma(t) = \sum_{j=1}^{n} \left[ a_j \cos\left(\frac{2\pi j t}{S}\right) + b_j \sin\left(\frac{2\pi j t}{S}\right) \right]$$

where $(a_j, b_j)$ are the Fourier coefficients (initial states).

#### Why ZeroSumNormal is PROBLEMATIC Here

**Key insight:** The Fourier basis with $j \geq 1$ **already excludes the DC component**.

For any harmonic $j \geq 1$:
$$\sum_{t=0}^{S-1} \cos\left(\frac{2\pi j t}{S}\right) = 0 \quad \text{and} \quad \sum_{t=0}^{S-1} \sin\left(\frac{2\pi j t}{S}\right) = 0$$

Therefore, **for ANY choice of coefficients**:
$$\sum_{t=0}^{S-1} \gamma(t) = 0 \quad \text{(automatic, no constraint needed)}$$

#### What ZeroSumNormal Actually Constrains

The `ZeroSumNormal` prior enforces a constraint on the **coefficients**, not the time-domain values:

$$\sum_{j=1}^{n} (a_j + b_j) = 0$$

By Riesz representation, this makes the seasonal effect orthogonal to:

$$g(t) = \sum_{j=1}^{n} \left[\cos\left(\frac{2\pi j t}{S}\right) + \sin\left(\frac{2\pi j t}{S}\right)\right]$$

**This is NOT a constant function!** It's a specific combination of harmonics with no physical interpretation.

#### Concrete Example: Single Harmonic ($n=1$)

With $n=1$, the constraint $a_1 + b_1 = 0$ means $a_1 = -b_1$, so:

$$\gamma(t) = a_1[\cos(\omega t) - \sin(\omega t)] = a_1\sqrt{2}\cos\left(\omega t + \frac{\pi}{4}\right)$$

**The phase is locked to $\pi/4$!** This is an arbitrary and severe restriction that prevents fitting seasonal patterns with other phases.

#### Closed Form for Saturated Case ($n = S/2$)

The "invisible" function $g(t)$ has the closed form:

$$g(t) = \begin{cases} S/2 & t = 0 \\ 0 & t \text{ even}, t \neq 0 \\ \cot\left(\frac{\pi t}{S}\right) - 1 & t \text{ odd} \end{cases}$$

The constraint enforces:

$$\frac{S}{2}\gamma(0) + \sum_{t \text{ odd}} \gamma(t)\left[\cot\left(\frac{\pi t}{S}\right) - 1\right] = 0$$

This has no natural physical interpretation.

#### Concrete "Invisible" Timeseries (S=12, Verified Numerically)

For monthly seasonality ($S=12$, $n=6$), the function $g(t)$ that **cannot be fit at all**:

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g(t) | +6.0 | +2.7 | 0 | 0 | 0 | -0.7 | 0 | -1.3 | 0 | -2.0 | 0 | -4.7 |

- **Coefficients of g(t):** all ones $(a_j = b_j = 1)$
- **Sum:** $\sum_j (a_j + b_j) = 12 \neq 0$ (violates constraint)
- **Projected coefficients:** all zeros!
- **Best fit:** $\gamma^*(t) = 0$ (the zero function)
- **R² = 0** (complete failure to fit)

This is a valid seasonal pattern that the constrained model **cannot represent at all**.

#### Likely Cause

This appears to be a **mistaken carryover** from time-domain seasonal models where `ZeroSumNormal` IS appropriate:

| Model Type | Parameterization | Zero-Sum Needed? |
|------------|------------------|------------------|
| Time-domain seasonal | Direct effects $s_1, s_2, \ldots, s_S$ | **Yes** (for identifiability vs intercept) |
| Fourier seasonal | Coefficients $a_1, b_1, \ldots, a_n, b_n$ | **No** (basis already excludes DC) |

#### Recommended Fix

Replace `ZeroSumNormal` with an unconstrained `Normal` prior:

```python
# Current (problematic)
_annual_seasonal = pm.ZeroSumNormal("params_freq", sigma=80, dims=annual_dims)

# Recommended fix
_annual_seasonal = pm.Normal("params_freq", mu=0, sigma=80, dims=annual_dims)
```

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

### For Time-Domain Seasonal Models (Cases 1 & 2)

**The original analysis is correct:**

1. **Yes**, there exists a function orthogonal to all zero-sum constrained functions: the constant function.

2. Properly specified models include an **intercept term** that captures the constant component.

3. The ZeroSumNormal constraint is an **identifiability constraint** that prevents the mean-seasonal ambiguity.

4. **Only if** you incorrectly omit the intercept would the constant component become "invisible."

### For CausalPy's StateSpaceTimeSeries (Case 3)

**⚠️ The `ZeroSumNormal` constraint on Fourier coefficients is likely a BUG:**

1. The Fourier basis with $j \geq 1$ **already excludes the DC component** — seasonal effects automatically average to zero over a complete cycle, regardless of coefficient values.

2. `ZeroSumNormal` constrains $\sum_j (a_j + b_j) = 0$, which is **not** equivalent to "seasonal averages to zero" — it's an arbitrary constraint in coefficient space.

3. This constraint **reduces model flexibility** unnecessarily:
   - For $n=1$: locks the phase to $\pi/4$
   - For general $n$: creates an "invisible" function with no physical interpretation

4. **Likely cause:** Mistaken carryover from time-domain seasonal models where zero-sum IS needed.

5. **Recommended fix:** Replace with `pm.Normal("params_freq", mu=0, sigma=80, dims=annual_dims)`

### Summary Table

| Context | Zero-Sum Constraint | Appropriate? |
|---------|---------------------|--------------|
| Time-domain seasonal effects $s_1, \ldots, s_S$ | $\sum_t s_t = 0$ | ✅ Yes (identifiability vs intercept) |
| Fourier coefficients $a_1, b_1, \ldots, a_n, b_n$ | $\sum_j (a_j + b_j) = 0$ | ❌ No (basis already excludes DC) |
