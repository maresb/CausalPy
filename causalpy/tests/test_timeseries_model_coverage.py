#   Copyright 2025 - 2026 The PyMC Labs Developers
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
"""
Tests for uncovered conditional logic in time series models.

This test file focuses on code coverage for edge cases and error handling
in BayesianBasisExpansionTimeSeries and StateSpaceTimeSeries.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import causalpy as cp


class MockComponent:
    """Mock component with apply method for testing custom components."""

    def apply(self, time_data):
        return time_data * 0


class MockComponentNoApply:
    """Mock component without apply method to test validation."""

    pass


class TestBayesianBasisExpansionTimeSeriesCoverage:
    """Test uncovered branches in BayesianBasisExpansionTimeSeries."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.randn(n_obs)

        X_da = xr.DataArray(
            np.zeros((n_obs, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return X_da, y_da

    def test_custom_trend_component_without_apply_method(self):
        """Test validation error when custom trend component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom trend_component must have an 'apply' method",
        ):
            cp.pymc_models.BayesianBasisExpansionTimeSeries(
                trend_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_seasonality_component_without_apply_method(self):
        """Test validation error when custom seasonality component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom seasonality_component must have an 'apply' method",
        ):
            cp.pymc_models.BayesianBasisExpansionTimeSeries(
                seasonality_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_components_with_apply_method(self, sample_data):
        """Test that custom components with apply method work."""
        X_da, y_da = sample_data

        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            trend_component=MockComponent(),
            seasonality_component=MockComponent(),
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Should not raise
        idata = model.fit(X_da, y_da)
        assert idata is not None

    def test_prepare_time_features_none_x(self):
        """Test error when X is None in _prepare_time_and_exog_features."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(ValueError, match="X cannot be None"):
            model._prepare_time_and_exog_features(None)

    def test_prepare_time_features_not_xarray(self):
        """Test error when X is not an xarray DataArray."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(TypeError, match="X must be an xarray DataArray"):
            model._prepare_time_and_exog_features(np.array([[1, 2, 3]]))

    def test_prepare_time_features_no_obs_ind_coord(self):
        """Test error when X lacks obs_ind coordinate."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(np.zeros((10, 0)), dims=["time", "coeffs"])

        with pytest.raises(ValueError, match="X must have 'obs_ind' coordinate"):
            model._prepare_time_and_exog_features(X_bad)

    def test_prepare_time_features_empty_obs_ind(self):
        """Test error when X has empty obs_ind."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(
            np.zeros((0, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": [], "coeffs": []},
        )

        with pytest.raises(ValueError, match="X must have at least one observation"):
            model._prepare_time_and_exog_features(X_bad)

    def test_prepare_time_features_non_datetime_obs_ind(self):
        """Test error when obs_ind doesn't contain datetime values."""
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        X_bad = xr.DataArray(
            np.zeros((10, 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": np.arange(10), "coeffs": []},
        )

        with pytest.raises(
            ValueError,
            match="X.coords\\['obs_ind'\\] must contain datetime values",
        ):
            model._prepare_time_and_exog_features(X_bad)

    def test_data_setter_error_x_mismatch(self, sample_data):
        """Test error when X exog var names don't match between fit and predict."""
        X_da, y_da = sample_data

        # Fit model without exogenous variables (empty X)
        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False}
        )
        model.fit(X_da, y_da)

        # Create X with exogenous variables for prediction
        dates_new = pd.date_range(start="2020-03-02", end="2020-03-10", freq="D")
        X_with_exog = xr.DataArray(
            np.random.randn(len(dates_new), 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates_new, "coeffs": ["x1"]},
        )

        # Should raise error about mismatch (model fit with [], trying to predict with ["x1"])
        with pytest.raises(
            ValueError,
            match="Exogenous variable names mismatch",
        ):
            model.predict(X_with_exog)

    def test_data_setter_error_missing_exog_vars(self, sample_data):
        """Test error when model expects exog vars but prediction X doesn't provide them."""
        X_da, y_da = sample_data
        dates = X_da.coords["obs_ind"].values

        # Create X with exogenous variables for fitting
        X_with_exog = xr.DataArray(
            np.random.randn(len(dates), 1),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": ["x1"]},
        )

        model = cp.pymc_models.BayesianBasisExpansionTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False}
        )
        model.fit(X_with_exog, y_da)

        # Try to predict with empty X
        dates_new = pd.date_range(start="2020-03-02", end="2020-03-10", freq="D")
        X_empty = xr.DataArray(
            np.zeros((len(dates_new), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates_new, "coeffs": []},
        )

        with pytest.raises(
            ValueError,
            match="Model was built with exogenous variables",
        ):
            model.predict(X_empty)


class TestStateSpaceTimeSeriesCoverage:
    """Test uncovered branches in StateSpaceTimeSeries."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2020-01-01", end="2020-02-01", freq="D")
        n_obs = len(dates)
        y_values = np.random.randn(n_obs) + 10

        y_da = xr.DataArray(
            y_values.reshape(-1, 1),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": dates, "treated_units": ["unit_0"]},
        )
        return y_da

    def test_custom_trend_component_without_apply_method(self):
        """Test validation error when custom trend component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom trend_component must have an 'apply' method",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                trend_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_custom_seasonality_component_without_apply_method(self):
        """Test validation error when custom seasonality component lacks apply method."""
        with pytest.raises(
            ValueError,
            match="Custom seasonality_component must have an 'apply' method",
        ):
            cp.pymc_models.StateSpaceTimeSeries(
                seasonality_component=MockComponentNoApply(),
                sample_kwargs={"draws": 10, "tune": 10, "progressbar": False},
            )

    def test_backwards_compatibility_coords_datetime_index(self, sample_data):
        """Test backwards compatibility with coords['datetime_index']."""
        y_da = sample_data
        dates = pd.DatetimeIndex(y_da.coords["obs_ind"].values)

        # Create y with integer obs_ind (old API)
        y_old_api = xr.DataArray(
            y_da.values,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(len(dates)), "treated_units": ["unit_0"]},
        )

        # Pass datetime via coords dict
        coords = {"datetime_index": dates}

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Should not raise - uses backwards compatibility path
        idata = model.fit(y=y_old_api, coords=coords)
        assert idata is not None

    def test_coords_datetime_index_not_datetimeindex(self, sample_data):
        """Test error when coords['datetime_index'] is not a DatetimeIndex."""
        y_da = sample_data
        n_obs = len(y_da)

        # Create y with integer obs_ind
        y_old_api = xr.DataArray(
            y_da.values,
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": np.arange(n_obs), "treated_units": ["unit_0"]},
        )

        # Pass non-DatetimeIndex via coords dict
        coords = {"datetime_index": np.arange(n_obs)}  # Not a DatetimeIndex!

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        with pytest.raises(
            ValueError,
            match="coords\\['datetime_index'\\] must be a pd.DatetimeIndex",
        ):
            model.fit(y=y_old_api, coords=coords)

    def test_build_model_y_none(self):
        """Test error when y is None in build_model."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError,
            match="y must be provided for StateSpaceTimeSeries.build_model",
        ):
            model.build_model(X=None, y=None)

    def test_build_model_y_no_obs_ind(self):
        """Test error when y lacks obs_ind coordinate."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        y_bad = xr.DataArray(
            np.random.randn(10, 1),
            dims=["time", "treated_units"],
            coords={"time": np.arange(10), "treated_units": ["unit_0"]},
        )

        with pytest.raises(ValueError, match="y must have 'obs_ind' coordinate"):
            model.build_model(y=y_bad)

    def test_build_model_y_empty_obs_ind(self):
        """Test error when y has empty obs_ind."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        y_bad = xr.DataArray(
            np.zeros((0, 1)),
            dims=["obs_ind", "treated_units"],
            coords={"obs_ind": [], "treated_units": ["unit_0"]},
        )

        with pytest.raises(ValueError, match="y must have at least one observation"):
            model.build_model(y=y_bad)

    def test_fit_y_none(self):
        """Test error when y is None in fit."""
        model = cp.pymc_models.StateSpaceTimeSeries(
            sample_kwargs={"draws": 10, "tune": 10, "progressbar": False}
        )

        with pytest.raises(
            ValueError,
            match="y must be provided for StateSpaceTimeSeries.fit",
        ):
            model.fit(y=None)

    def test_predict_out_of_sample_x_none(self, sample_data):
        """Test error when X is None for out-of-sample predictions."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Create dummy X for fit (state-space doesn't use it)
        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        with pytest.raises(
            ValueError,
            match="X must be provided for out-of-sample predictions",
        ):
            model.predict(X=None, out_of_sample=True)

    def test_predict_out_of_sample_x_no_coords(self, sample_data):
        """Test error when X lacks coords for out-of-sample predictions."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        # Fit model
        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        # Try to predict with numpy array (no coords)
        X_no_coords = np.zeros((5, 0))

        with pytest.raises(
            ValueError,
            match="X must have 'obs_ind' coordinate with datetime values",
        ):
            model.predict(X=X_no_coords, out_of_sample=True)

    def test_score_y_none(self, sample_data):
        """Test error when y is None in score."""
        y_da = sample_data

        model = cp.pymc_models.StateSpaceTimeSeries(
            level_order=1,
            seasonal_length=7,
            sample_kwargs={"draws": 10, "tune": 10, "chains": 1, "progressbar": False},
        )

        dates = y_da.coords["obs_ind"].values
        dummy_X = xr.DataArray(
            np.zeros((len(dates), 0)),
            dims=["obs_ind", "coeffs"],
            coords={"obs_ind": dates, "coeffs": []},
        )
        model.fit(X=dummy_X, y=y_da)

        # StateSpaceTimeSeries.score calls super().score() which doesn't validate y
        # So it raises AttributeError when trying to call y.sel()
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            model.score(X=dummy_X, y=None)


class TestZeroSumNormalConstraintAnalysis:
    """
    Analyze the ZeroSumNormal constraint on FrequencySeasonality parameters.

    IMPORTANT: The ZeroSumNormal constraint on Fourier coefficients is problematic.

    Key facts:
    1. FrequencySeasonality uses harmonics j=1,2,...,n (NOT j=0)
       - Array index 0 ("Cos_0_freq") corresponds to j=1, not j=0
       - There is NO constant (DC) term in the basis
    2. The seasonal effect already has zero mean (automatic with j≥1)
    3. The ZeroSumNormal constraint Σ(a_j + b_j) = 0 is an arbitrary
       restriction with no natural physical interpretation
    4. For n=1, it restricts the phase angle to π/4 + kπ
    """

    def test_fourier_basis_starts_at_j1_not_j0(self):
        """
        Verify that FrequencySeasonality uses j=1,2,... (not j=0).

        The array index "Cos_0" means the 0th element in the array,
        which corresponds to frequency j=1, NOT the DC component.
        """
        S = 12  # period
        n = S // 2  # number of harmonics

        def build_correct_fourier_basis(t, S, n):
            """Build Fourier basis matching FrequencySeasonality (j starts at 1)."""
            basis = []
            for j in range(1, n + 1):  # j = 1, 2, ..., n
                basis.append(np.cos(2 * np.pi * j * t / S))
                if j < n:  # No sin at Nyquist frequency
                    basis.append(np.sin(2 * np.pi * j * t / S))
            return np.column_stack(basis)

        t = np.arange(S)
        X = build_correct_fourier_basis(t, S, n)

        # The first column should be cos(2π·1·t/S), NOT cos(0)=1
        cos_j1 = np.cos(2 * np.pi * 1 * t / S)
        assert np.allclose(X[:, 0], cos_j1), (
            "First column should be cos(2π·1·t/S), not constant"
        )

        # The second column should be sin(2π·1·t/S), NOT sin(0)=0
        sin_j1 = np.sin(2 * np.pi * 1 * t / S)
        assert np.allclose(X[:, 1], sin_j1), (
            "Second column should be sin(2π·1·t/S), not zero"
        )
        assert not np.allclose(X[:, 1], 0), "sin(2π·1·t/S) should NOT be zero!"

    def test_seasonal_already_has_zero_mean(self):
        """
        Verify that the Fourier basis with j≥1 already produces zero-mean signals.

        The ZeroSumNormal constraint is NOT needed for zero-mean seasonality.
        """
        S = 12
        n = S // 2
        t = np.arange(S)

        # Any linear combination of cos/sin with j≥1 has zero mean
        for j in range(1, n + 1):
            cos_j = np.cos(2 * np.pi * j * t / S)
            sin_j = np.sin(2 * np.pi * j * t / S)
            assert np.isclose(np.mean(cos_j), 0, atol=1e-10), (
                f"cos(2π·{j}·t/S) should have zero mean"
            )
            assert np.isclose(np.mean(sin_j), 0, atol=1e-10), (
                f"sin(2π·{j}·t/S) should have zero mean"
            )

    def test_zero_sum_constraint_restricts_phase_for_n1(self):
        """
        Demonstrate that ZeroSumNormal restricts phase when n=1.

        For a single harmonic: γ(t) = a·cos(ωt) + b·sin(ωt)
        The constraint a + b = 0 means b = -a
        So γ(t) = a·[cos(ωt) - sin(ωt)] = a·√2·cos(ωt + π/4)

        This restricts the phase to π/4 (modulo π), losing flexibility.
        """
        S = 12
        omega = 2 * np.pi / S
        t = np.arange(S)

        # With zero-sum constraint (a + b = 0), we can only fit phases = π/4 + kπ
        a = 1.0
        b = -a  # forced by constraint

        # The constrained signal
        y_constrained = a * np.cos(omega * t) + b * np.sin(omega * t)

        # This is equivalent to cos(ωt + π/4) scaled by √2
        y_equivalent = a * np.sqrt(2) * np.cos(omega * t + np.pi / 4)
        assert np.allclose(y_constrained, y_equivalent)

        # A signal with phase=0 CANNOT be perfectly represented with zero-sum
        # constraint. To fit y = cos(ωt), we need a=1, b=0, but a+b=1≠0!
        # The constraint loses one degree of freedom.

    def test_orthogonal_function_closed_form(self):
        """
        Verify the closed-form formula for the function orthogonal to zero-sum.

        The ZeroSumNormal constraint enforces orthogonality to:
        g(t) = Σⱼ[cos(2πjt/S) + sin(2πjt/S)] for j=1,...,n

        For saturated case (n=S/2):
        g(t) = S/2       if t = 0
             = 0         if t even, t ≠ 0
             = cot(πt/S) - 1  if t odd
        """
        S = 12
        n = S // 2
        t_vals = np.arange(S)

        def g_direct(t, S, n):
            """Direct computation."""
            return sum(
                np.cos(2 * np.pi * j * t / S) + np.sin(2 * np.pi * j * t / S)
                for j in range(1, n + 1)
            )

        def g_closed_form(t, S):
            """Closed-form for saturated case n=S/2."""
            t = t % S
            if t == 0:
                return S / 2
            elif t % 2 == 0:
                return 0
            else:
                return 1 / np.tan(np.pi * t / S) - 1

        for t in t_vals:
            direct = g_direct(t, S, n)
            closed = g_closed_form(t, S)
            assert np.isclose(direct, closed, atol=1e-10), (
                f"Closed-form mismatch at t={t}: {direct} vs {closed}"
            )

    def test_unrepresentable_signal_cannot_be_fitted(self):
        """
        Verify that the signal g(t) with all Fourier coefficients = 1
        CANNOT be fitted by the ZeroSumNormal-constrained model.

        This demonstrates a concrete failure mode of the constraint.
        """
        S = 12
        n = S // 2  # 6 harmonics

        # Build basis matching pymc-extras FrequencySeasonality
        # j=1..5: cos + sin pairs; j=6: cos only (sin at Nyquist excluded)
        def build_basis(t, S):
            basis = []
            for j in range(1, n):  # j = 1 to 5
                basis.append(np.cos(2 * np.pi * j * t / S))
                basis.append(np.sin(2 * np.pi * j * t / S))
            basis.append(np.cos(2 * np.pi * n * t / S))  # Nyquist cos
            return np.column_stack(basis)

        t = np.arange(S)
        X = build_basis(t, S)
        n_params = X.shape[1]  # 11

        # The unrepresentable signal: θ = (1, 1, ..., 1)
        theta_invisible = np.ones(n_params)
        g = X @ theta_invisible

        # Verify g has zero mean (it's a valid seasonal signal)
        assert np.isclose(np.mean(g), 0, atol=1e-10), "g should have zero mean"

        # Unconstrained OLS recovers θ exactly
        theta_ols, _, _, _ = np.linalg.lstsq(X, g, rcond=None)
        assert np.allclose(theta_ols, 1.0), "OLS should recover θ = (1,1,...,1)"

        # Zero-sum projection kills the entire signal
        theta_zs = theta_ols - np.mean(theta_ols)
        assert np.isclose(np.sum(theta_zs), 0), "θ_zs should sum to zero"
        assert np.allclose(theta_zs, 0), "θ_zs should be all zeros"

        # The fitted signal is identically zero
        y_fit = X @ theta_zs
        assert np.allclose(y_fit, 0), "Fitted signal should be zero"

        # 100% of variance is unexplained
        residual = g - y_fit
        assert np.allclose(residual, g), "Residual should equal original signal"

        # The signal variance is non-trivial
        assert np.var(g) > 1, f"Signal should have significant variance: {np.var(g)}"
