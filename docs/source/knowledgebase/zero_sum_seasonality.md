# Zero-sum seasonality and identifiability in StateSpaceTimeSeries

This note explains why `StateSpaceTimeSeries` uses `pm.ZeroSumNormal` for the seasonal parameters and why this does not remove any fit capacity when a level/trend component is present.

## Where the constraint appears

In `causalpy/pymc_models.py`, the state-space model builds a level/trend component and a frequency seasonality component, then places a `ZeroSumNormal` prior on the seasonal parameter vector (`params_freq`). This enforces that the seasonal state sums to zero over one seasonal cycle.

## Orthogonal complement via Riesz

Let S be the subspace of seasonal functions with zero mean over a full period T:

S = { f in L2[0,T] : \int_0^T f(t) dt = 0 }.

Use the L2 inner product <f,g> = \int_0^T f(t) g(t) dt. Define the linear functional L(f) = \int_0^T f(t) dt. By the Riesz representation theorem there exists a unique g in L2 such that L(f) = <f,g> for all f. Here g(t) = 1 because <f,1> = \int_0^T f(t) dt. Therefore S = ker L and S^\perp = span{1}. In Fourier language this is the k=0 (constant) mode; all k != 0 modes have zero mean.

For a discrete season of length m, the same reasoning uses <x,y> = sum_{i=1}^m x_i y_i. The zero-sum subspace {x : sum x_i = 0} has orthogonal complement span{(1,1,...,1)}.

## What this means for model fit

The only component orthogonal to the zero-sum seasonal subspace is the constant (mean) component. In the state-space model, that constant is absorbed by the level/trend component. Any seasonal series y_t over a period can be decomposed as

y_t = \bar{y} + (y_t - \bar{y}),

where \bar{y} is the mean over the period and the residual term has zero sum. The zero-sum seasonal state captures the residual, and the level/trend captures \bar{y}. This is a standard identifiability constraint: it prevents the seasonal state and level from both representing the same constant offset.

## When would something be "invisible"?

If you removed the level/trend component entirely and only kept the zero-sum seasonal state, then a pure constant series would be unidentifiable because it lives in the orthogonal complement. However, `StateSpaceTimeSeries` always includes a level/trend component (via `LevelTrendComponent` by default), so that constant mode is still fully fit. There is no invisible time series in the default model; the constraint only fixes the seasonal mean at zero so the level can take it.
