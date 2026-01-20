# Sum-to-zero constraints, orthogonality, and “invisible” components

This note addresses a common confusion around using a *sum-to-zero* constraint (sometimes implemented as `ZeroSumNormal`) for seasonal/categorical effects.

## The zero-sum subspace and its orthogonal complement (Riesz viewpoint)

Let \(K \ge 2\) and consider vectors \(s \in \mathbb{R}^K\) with the standard inner product
\[
\langle a, b \rangle \;=\; \sum_{k=1}^K a_k b_k.
\]

Define the *zero-sum* (a.k.a. “sum-to-zero”) subspace
\[
S \;=\; \{ s \in \mathbb{R}^K : \mathbf{1}^\top s = 0 \},
\]
where \(\mathbf{1} = (1,1,\dots,1)^\top\).

Now consider the linear functional \(L(s) = \mathbf{1}^\top s = \sum_k s_k\). By the Riesz representation theorem (in finite-dimensional Euclidean space), there exists a unique vector \(v\) such that
\[
L(s) \;=\; \langle v, s \rangle \quad \text{for all } s.
\]
Taking \(v = \mathbf{1}\) gives exactly \(L(s)=\langle \mathbf{1}, s\rangle\), so the Riesz representer of “sum” is \(\mathbf{1}\).

Because \(S = \ker(L)\), the orthogonal complement is
\[
S^\perp \;=\; \mathrm{span}\{\mathbf{1}\}.
\]

### Closed form “orthogonal function”

Any nonzero scalar multiple of the constant vector is orthogonal to every zero-sum vector. A particularly simple normalized choice is
\[
f_\perp \;=\; \frac{1}{\sqrt{K}} \mathbf{1},
\]
which satisfies \(\langle f_\perp, s\rangle = 0\) for all \(s \in S\).

## Trigonometric / exponential viewpoint (Fourier “DC component”)

If you think of a seasonal effect as a discrete periodic function on \(K\) points, a sum-to-zero constraint removes the *zero-frequency* component.

- In a trigonometric Fourier series, it removes the constant term \(a_0\).
- In the complex exponential form, it removes the \(k=0\) Fourier coefficient.

That “DC component” is exactly the same direction as \(\mathbf{1}\) above.

## What this means in a regression / time-series model

Seasonality/categorical effects are often used via a mapping
\[
y_t \approx \alpha + s_{c(t)} + \varepsilon_t,
\]
where \(c(t)\in\{1,\dots,K\}\) is the category at time \(t\) (month-of-year, day-of-week, etc.).

If you constrain \(s \in S\) (sum-to-zero), then **the only thing you removed is the ability of \(s\) itself to carry a constant offset**. That constant offset is instead carried by the intercept \(\alpha\).

This is an identifiability fix:
- Without the constraint, \((\alpha, s)\) is not unique because \((\alpha + \delta, s - \delta \mathbf{1})\) yields the same \(y_t\).
- With the constraint \(\mathbf{1}^\top s = 0\), the decomposition becomes unique and well-identified.

## When you really do get an “unfit-able / invisible” component

If you **omit the intercept** *and* require \(\mathbf{1}^\top s = 0\), then a constant offset in the data cannot be represented by the model.

Equivalently: the fitted values live in a subspace orthogonal to the all-ones vector, so the constant component of the signal is *guaranteed* to remain in the residual.

See `causalpy/tests/test_zero_sum_constraints.py` for a concrete construction and verification.

