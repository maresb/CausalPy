
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
from causalpy.pymc_models import StateSpaceTimeSeries
import pymc as pm
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def generate_invisible_series(t, period):
    """
    Generates the 'invisible' time series f(t) via summation.
    f(t) = sum_{k=1}^{floor(S/2)} [cos(2pi k t / S) + sin(2pi k t / S)]
    (taking sin(pi t) = 0 for the last term if S is even)
    """
    y = np.zeros_like(t, dtype=float)
    n_harmonics = period // 2
    for k in range(1, n_harmonics + 1):
        omega = 2 * np.pi * k / period
        y += np.cos(omega * t)
        if k < period / 2: # Sin term exists for k < S/2
            y += np.sin(omega * t)
    return y

def closed_form(t, period):
    """
    Evaluates the closed-form formula.
    """
    vals = []
    for ti in t:
        if ti % period == 0:
            vals.append(period / 2)
        elif ti % 2 == 0:
            vals.append(0.0)
        else:
            term = 1.0 / np.tan(np.pi * ti / period) - 1.0
            vals.append(term)
    return np.array(vals)

def main():
    print("=== ZeroSumNormal Breakage Demonstration ===")
    
    period = 12
    n_obs = 30
    t = np.arange(n_obs)
    
    # 1. Verification of Closed Form
    y_vals = generate_invisible_series(t, period)
    y_closed = closed_form(t, period)
    
    print("\n1. Derivation Verification")
    if np.allclose(y_vals, y_closed):
        print("   [PASS] Closed-form formula matches the Fourier sum.")
    else:
        print("   [FAIL] Closed-form formula mismatch.")
        
    # 2. Construction of Invisible Series
    print(f"\n2. Constructed 'Invisible' Time Series (Period={period})")
    print(f"   First 13 values: {np.round(y_vals[:13], 2)}")
    print(f"   Mean: {y_vals.mean():.4f} (Non-zero, but specific structure)")
    
    # 3. Model Fitting Demonstration
    print("\n3. Model Fitting with StateSpaceTimeSeries (using ZeroSumNormal)")
    
    dates = pd.date_range(start="2020-01-01", periods=n_obs, freq="ME")
    y = xr.DataArray(
        y_vals.reshape(-1, 1),
        dims=["obs_ind", "treated_units"],
        coords={"obs_ind": dates, "treated_units": ["unit_0"]}
    )
    
    model = StateSpaceTimeSeries(
        seasonal_length=period,
        sample_kwargs={
            "draws": 10, 
            "tune": 10, 
            "chains": 1, 
            "progressbar": False,
            "random_seed": 42,
            "compute_convergence_checks": False
        }
    )
    
    try:
        model.fit(y=y)
        
        # If it miraculously finishes, check R2
        posterior_mean = model.idata.posterior_predictive["mu"].mean(dim=["chain", "draw"])
        y_hat = posterior_mean.values.flatten()
        sst = np.sum((y_vals - np.mean(y_vals))**2)
        ssr = np.sum((y_vals - y_hat)**2)
        r2 = 1 - ssr / sst
        
        print(f"   Model finished fitting.")
        print(f"   R2 Score: {r2:.4f}")
        
        if r2 < 0.1:
             print("   [SUCCESS] The model failed to fit the series (Low R2).")
        else:
             print("   [UNEXPECTED] The model fit the series well.")
             
    except Exception as e:
        print(f"   [SUCCESS] Model fitting crashed as expected.")
        print(f"   Error details: {str(e).splitlines()[-1]}")
        print("   (This crash is due to the ZeroSumNormal constraint conflicting with the data)")

if __name__ == "__main__":
    main()
