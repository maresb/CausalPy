# Zero-sum seasonality and the “missing” (orthogonal) direction

Historically, `StateSpaceTimeSeries` used a zero-sum prior for the frequency-seasonality *initial state parameters*:

- the model builds a `pymc-extras` structural state-space model with a frequency seasonality component
- the parameter vector named `"params_freq"` is given a `pm.ZeroSumNormal(...)` prior

This can look odd at first glance, and (for frequency-domain seasonality) it is indeed **not** the usual “sum-to-zero over the seasonal cycle” identifiability constraint.

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

## Important distinction: time-domain vs frequency-domain seasonality

In **time-domain** seasonal models with \(s\) seasonal effects (e.g. one effect per month), the raw parameterization includes a constant/mean direction unless you constrain it. In that setting, “sum-to-zero over a cycle” is a standard identifiability constraint.

In **frequency-domain** seasonal models like `pymc-extras` `FrequencySeasonality`, the seasonal signal is represented using Fourier modes with \(j \ge 1\). Those basis functions have zero average over a complete cycle, so the constant (DC) component is excluded by construction.

That means a separate zero-sum constraint on the Fourier initial-state parameters is not needed to keep the seasonal component mean-zero over a cycle, and it can impose an arbitrary restriction (e.g. when \(n=1\), it enforces a linear relationship between the sine/cosine initial states, which effectively fixes the phase).

## Identifiability with a level/trend (what *can* go wrong)

If your observation model is (schematically)

\[
y_t = \ell_t + s_t + \varepsilon_t,
\]

and the seasonality \(s_t\) is allowed to include an arbitrary constant offset, then there is a non-identifiability:

- for any constant \(c\), the transformed pair
  - \(\ell_t' = \ell_t + c\)
  - \(s_t' = s_t - c\)
  gives the **same** \(\ell_t' + s_t' = \ell_t + s_t\), hence the same likelihood.

Constraining a **time-domain** seasonal component to have **zero mean over one period** removes exactly that redundant “constant” degree of freedom, forcing the overall mean level to be represented by the level/trend component instead.

That is precisely why a zero-sum constraint is commonly used in structural time-series models.

## When an “invisible” series really exists

An “invisible” series appears only if you **remove** any intercept/level/trend term and try to explain data using *only* a zero-sum seasonal component.

In that case, the constant direction (the orthogonal complement described above) cannot be represented:

- a pure zero-sum seasonal component has mean zero over each seasonal cycle
- therefore it cannot equal a non-zero constant time series

So, if you fit a constant series \(y_t \equiv c\) with a seasonal-only, zero-sum-constrained model, the best fit cannot drive the residuals to zero; the constant part is truly “invisible” to the seasonal subspace.

In CausalPy, `StateSpaceTimeSeries` includes a level/trend component by default (`LevelTrendComponent`), and `FrequencySeasonality` excludes the DC component by construction, so the constant direction is not “invisible” in the default model.
