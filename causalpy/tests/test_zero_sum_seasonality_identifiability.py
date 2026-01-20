#   Copyright 2026 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import xarray as xr
from pymc.distributions.multivariate import ZeroSumNormalRV

from causalpy.pymc_models import StateSpaceTimeSeries


def _zero_sum_seasonal_design_matrix(*, n_obs: int, period: int) -> np.ndarray:
    """
    Create a seasonal design matrix with a sum-to-zero constraint.

    Parameterization:
    - free parameters: theta in R^(period-1)
    - seasonal effects: beta_j = theta_j for j=0..period-2
                      : beta_(period-1) = -sum(theta)

    Then y_hat_t = beta_{season(t)} = Z[t] @ theta.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if period < 2:
        raise ValueError("period must be at least 2")

    z = np.zeros((n_obs, period - 1), dtype=float)
    for t in range(n_obs):
        season = t % period
        if season < period - 1:
            z[t, season] = 1.0
        else:
            z[t, :] = -1.0
    return z


def test_zero_sum_subspace_orthogonal_complement_is_constant_vector() -> None:
    m = 12
    ones = np.ones(m)

    # Any zero-sum vector is orthogonal to the constant vector.
    rng = np.random.default_rng(0)
    x = rng.normal(size=m)
    x = x - x.mean()  # enforce sum(x)=0
    assert abs(x.sum()) < 1e-12
    assert abs(x @ ones) < 1e-12

    # Conversely, if v is orthogonal to *all* zero-sum vectors, then v must be constant.
    #
    # A convenient spanning set for {x: sum(x)=0} is {e_i - e_{m-1}} for i=0..m-2.
    v = rng.normal(size=m)
    for i in range(m - 1):
        basis_vec = np.zeros(m)
        basis_vec[i] = 1.0
        basis_vec[m - 1] = -1.0
        # enforce v is orthogonal to each basis vector
        v[i] = v[m - 1]
        assert abs(basis_vec @ v) < 1e-12

    assert np.allclose(v, v[0] * ones)


def test_seasonal_only_zero_sum_model_cannot_fit_constant_series() -> None:
    period = 12
    n_cycles = 8
    n_obs = period * n_cycles  # balanced across seasons

    z = _zero_sum_seasonal_design_matrix(n_obs=n_obs, period=period)

    c = 3.25
    y = np.full(n_obs, c)

    # Least squares best-fit under the zero-sum seasonal parameterization.
    theta_hat, *_ = np.linalg.lstsq(z, y, rcond=None)
    y_hat = z @ theta_hat

    # Every prediction from this model has mean zero across a full period (hence over this balanced sample).
    assert abs(y_hat.mean()) < 1e-12

    # Therefore it cannot match a non-zero constant series.
    mse = float(np.mean((y_hat - y) ** 2))
    assert mse > 0.1  # far from numerical noise
    assert np.isclose(mse, c**2, atol=1e-8)


def test_seasonal_plus_intercept_can_fit_constant_series() -> None:
    period = 12
    n_cycles = 8
    n_obs = period * n_cycles

    z = _zero_sum_seasonal_design_matrix(n_obs=n_obs, period=period)
    x = np.column_stack([np.ones(n_obs), z])

    c = 3.25
    y = np.full(n_obs, c)

    coef_hat, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef_hat

    assert np.allclose(y_hat, y)
    assert np.isclose(
        coef_hat[0], c, atol=1e-10
    )  # intercept captures the constant direction


def test_state_space_timeseries_default_components_import() -> None:
    # This primarily guards against pymc-extras import path changes.
    model = StateSpaceTimeSeries(level_order=2, seasonal_length=12)
    assert model._get_trend_component() is not None
    assert model._get_seasonality_component() is not None


def test_state_space_timeseries_params_freq_is_not_zero_sum_normal() -> None:
    # FrequencySeasonality should not need an extra sum-to-zero constraint on its initial state parameters.
    n_obs = 24
    obs_ind = np.arange(n_obs)
    y = xr.DataArray(
        np.zeros((n_obs, 1)),
        dims=["obs_ind", "treated_units"],
        coords={
            "obs_ind": np.array(
                np.datetime64("2020-01-01") + obs_ind.astype("timedelta64[D]")
            ),
            "treated_units": ["unit_0"],
        },
    )

    m = StateSpaceTimeSeries(
        level_order=2,
        seasonal_length=12,
        sample_kwargs={"draws": 1, "tune": 1, "chains": 1},
    )
    m.build_model(y=y, coords=None)
    assert m.second_model is not None
    rv = m.second_model["params_freq"]
    assert rv.owner is not None
    assert type(rv.owner.op) is not ZeroSumNormalRV
