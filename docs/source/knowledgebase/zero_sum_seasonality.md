# Zero-sum seasonality and the “missing” (orthogonal) direction

`StateSpaceTimeSeries` uses a zero-sum prior for the seasonal parameters:

- the model builds a `pymc-extras` structural state-space model with a frequency seasonality component
- the parameter vector named `"params_freq"` is given a `pm.ZeroSumNormal(...)` prior

This can look odd at first glance, but it is a standard identifiability constraint: it prevents the seasonal component from “stealing” the overall mean level (intercept) from the trend/level component.

## The orthogonal complement in closed form (Riesz representation)

### Continuous-time version (intrinsically real; no sums)

Let \(H = L^2([0,T])\) with inner product

\[
\langle f, g \rangle = \int_0^T f(t)\, g(t)\, dt.
\]

Consider the linear functional \(L : H \to \mathbb{R}\) given by

\[
L(f) = \int_0^T f(t)\, dt.
\]

By the Riesz representation theorem, there exists a unique \(h \in H\) such that \(L(f) = \langle f, h \rangle\) for all \(f \in H\). Here,

\[
L(f) = \int_0^T f(t)\, dt = \int_0^T f(t)\cdot 1\, dt = \langle f, 1 \rangle,
\]

so the Riesz representer is the constant function

\[
h(t) \equiv 1 \quad (\text{or normalized as } 1/\sqrt{T}).
\]

The “zero-mean” (a.k.a. zero-sum / integral-zero) subspace is

\[
S = \{ f \in H : L(f) = 0 \} = \{ f : \int_0^T f(t)\,dt = 0 \} = h^\perp,
\]

and therefore its orthogonal complement is

\[
S^\perp = \text{span}\{h\} = \text{span}\{1\}.
\]

In words: **the simplest nonzero function orthogonal to all zero-mean seasonal functions is just the constant function**.

### Discrete seasonal version (the one used in many seasonal state-space models)

Let \(H = \mathbb{R}^m\) with dot product \(\langle a,b\rangle = a^\top b\). Define the zero-sum subspace

\[
S = \left\{ s \in \mathbb{R}^m : \sum_{j=1}^m s_j = 0 \right\}.
\]

Let \(\mathbf{1} = (1,1,\ldots,1)^\top\). Then

\[
s \in S \iff \mathbf{1}^\top s = 0 \iff \langle s, \mathbf{1}\rangle = 0,
\]

so again \(S = \mathbf{1}^\perp\) and \(S^\perp = \text{span}\{\mathbf{1}\}\).

## Why this is usually *good*: identifiability with a level/trend

Seasonal components and intercept/level components overlap in exactly the constant direction.

If your observation model is (schematically)

\[
y_t = \ell_t + s_t + \varepsilon_t,
\]

and the seasonality \(s_t\) is allowed to include an arbitrary constant offset, then there is a non-identifiability:

- for any constant \(c\), the transformed pair
  - \(\ell_t' = \ell_t + c\)
  - \(s_t' = s_t - c\)
  gives the **same** \(\ell_t' + s_t' = \ell_t + s_t\), hence the same likelihood.

Constraining the seasonal component to have **zero mean over one period** removes exactly that redundant “constant” degree of freedom, forcing the overall mean level to be represented by the level/trend component instead.

That is precisely why a zero-sum constraint is commonly used in structural time-series models.

## When an “invisible” series really exists

An “invisible” series appears only if you **remove** any intercept/level/trend term and try to explain data using *only* a zero-sum seasonal component.

In that case, the constant direction (the orthogonal complement described above) cannot be represented:

- a pure zero-sum seasonal component has mean zero over each seasonal cycle
- therefore it cannot equal a non-zero constant time series

So, if you fit a constant series \(y_t \equiv c\) with a seasonal-only, zero-sum-constrained model, the best fit cannot drive the residuals to zero; the constant part is truly “invisible” to the seasonal subspace.

In CausalPy, `StateSpaceTimeSeries` includes a level/trend component by default (`LevelTrendComponent`), so the constant direction is **not** invisible: it is captured by the level/trend, while the seasonal component is kept mean-zero for identifiability.

