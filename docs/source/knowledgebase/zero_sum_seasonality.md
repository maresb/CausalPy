# Zero-sum seasonality and the "missing" component

This note clarifies how zero-sum seasonal constraints (such as
`pm.ZeroSumNormal`) interact with identifiability. It also explains the
orthogonal complement and when a component becomes unidentifiable.

At the time of writing, this repository does not define
`_annual_seasonal` or use `ZeroSumNormal` directly. The discussion below
applies to any model that uses a zero-sum seasonal effect.

## The space of zero-sum seasonal effects

Let `s` in `R^m` represent seasonal effects over `m` seasons (for example,
12 months). The zero-sum constraint is:

`sum_{i=1}^m s_i = 0`.

Define the inner product `<a, b> = sum_i a_i b_i` (or any constant
multiple of that). The linear functional `L(s) = sum_i s_i` has a Riesz
representer `g` such that:

`L(s) = <s, g>`.

The representer is `g = (1, 1, ..., 1)`, so the orthogonal complement of
the zero-sum subspace is the span of the all-ones vector. In other words,
the only function orthogonal to all zero-sum seasonal effects is a
constant function.

For periodic functions in `L2[0, T]`, the same statement is:

`H = {f : integral_0^T f(t) dt = 0}`, and `H^perp = span{1}`.

In Fourier form,

`f(t) = a0 + sum_{k>=1} (a_k cos(k omega t) + b_k sin(k omega t))`,

the zero-sum (zero-mean) constraint sets `a0 = 0`. The orthogonal
component is exactly the `k = 0` term, i.e., the constant function. In
the complex exponential basis, this is the `e^{i 0 omega t}` term.

## What this means for modeling

A model that uses a zero-sum seasonal effect but also includes an
intercept (or a trend with a free offset) can represent any seasonal
function:

`s(t) = mean(s) * 1 + (s(t) - mean(s))`.

The first term is the constant component handled by the intercept; the
second term is zero-sum and handled by the seasonal effect. With an
intercept, there is no "invisible" time series component. The constraint
only prevents the seasonal effect from absorbing the global level.

If you omit the intercept, the constant component becomes unidentifiable.
That constant is the only "invisible" time series.

## Toy verification (no intercept)

Consider a constant series `y = c * 1`. The best zero-sum fit is the
zero vector, so the residual remains constant.

```
import numpy as np

m = 12
y = np.full(m, 5.0)
P = np.eye(m) - np.ones((m, m)) / m  # projection onto zero-sum subspace
s_hat = P @ y
residual = y - s_hat

print(s_hat)     # all zeros
print(residual)  # still all 5s
```

Any method that only allows zero-sum seasonal effects cannot reduce the
constant residual. Add an intercept and the constant residual disappears.
