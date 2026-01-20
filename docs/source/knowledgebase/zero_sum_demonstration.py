"""
Demonstration: Orthogonal Functions to Zero-Sum Constrained Seasonal Effects

This script verifies the mathematical analysis showing that:
1. The orthogonal complement of zero-sum constrained functions is the constant function
2. A model with ONLY ZeroSumNormal (no intercept) cannot fit constant signals
3. A model WITH intercept + ZeroSumNormal can fit all signals correctly

Run with: pixi run python docs/source/knowledgebase/zero_sum_demonstration.py
"""

import warnings
from typing import Any

import arviz as az
import numpy as np
import pymc as pm

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def create_seasonal_data(
    n_periods: int = 12,
    n_cycles: int = 4,
    constant: float = 10.0,
    seasonal_amplitude: float = 2.0,
    noise_std: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic seasonal time series data.
    
    Returns:
        t: time indices
        y: observed values
        true_seasonal: the true seasonal component (sums to zero)
    """
    n_total = n_periods * n_cycles
    t = np.arange(n_total)
    period_index = t % n_periods
    
    # Create seasonal pattern that sums to zero
    raw_seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_periods) / n_periods)
    true_seasonal = raw_seasonal - raw_seasonal.mean()  # Ensure zero-sum
    
    # Build the time series: constant + seasonal + noise
    y = constant + true_seasonal[period_index] + rng.normal(0, noise_std, n_total)
    
    return t, y, true_seasonal


def verify_orthogonality():
    """
    Verify that constant vectors are orthogonal to the zero-sum subspace.
    """
    print("=" * 70)
    print("PART 1: Mathematical Verification of Orthogonality")
    print("=" * 70)
    
    n = 12  # e.g., 12 months
    
    # The constant vector
    ones = np.ones(n)
    
    # Create several zero-sum vectors
    zero_sum_vectors = [
        np.sin(2 * np.pi * np.arange(n) / n),  # First harmonic
        np.cos(2 * np.pi * np.arange(n) / n),  # First harmonic (cos)
        np.sin(4 * np.pi * np.arange(n) / n),  # Second harmonic
        rng.normal(0, 1, n),  # Random vector
    ]
    
    # Center random vector to have zero sum
    zero_sum_vectors[-1] -= zero_sum_vectors[-1].mean()
    
    print(f"\nConstant vector: 1 = ({', '.join(['1']*min(n,5))}, ...)")
    print(f"\nVerifying orthogonality (inner products with constant vector):\n")
    
    for i, v in enumerate(zero_sum_vectors):
        inner_product = np.dot(ones, v)
        vector_sum = np.sum(v)
        print(f"  Vector {i+1}: sum = {vector_sum:+.6f}, <1, v> = {inner_product:+.6f}")
    
    print(f"\nConclusion: All zero-sum vectors are orthogonal to constant vectors.")
    print(f"The inner product <1, s> = sum(s) = 0 for any zero-sum s.")


def fourier_analysis():
    """
    Demonstrate the Fourier perspective on zero-sum constraints.
    """
    print("\n" + "=" * 70)
    print("PART 2: Fourier Analysis of Zero-Sum Constraint")
    print("=" * 70)
    
    n = 12
    
    # A general signal
    signal = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / n) + rng.normal(0, 0.1, n)
    
    # Its Fourier transform
    fft_signal = np.fft.fft(signal)
    
    print(f"\nOriginal signal mean: {signal.mean():.4f}")
    print(f"DC component (k=0): {fft_signal[0].real/n:.4f}")
    
    # Zero-sum version (remove DC component)
    zero_sum_signal = signal - signal.mean()
    fft_zero_sum = np.fft.fft(zero_sum_signal)
    
    print(f"\nAfter zero-sum constraint:")
    print(f"  Signal mean: {zero_sum_signal.mean():.10f}")
    print(f"  DC component (k=0): {fft_zero_sum[0].real/n:.10f}")
    
    print(f"\nConclusion: Zero-sum constraint ⟺ DC component = 0")
    print(f"The 'invisible' signal is the pure DC component (constant function).")


def model_without_intercept(y: np.ndarray, period_index: np.ndarray, n_periods: int) -> dict[str, Any]:
    """
    Fit a model with ONLY ZeroSumNormal seasonal component (NO intercept).
    This model cannot fit the constant component.
    """
    with pm.Model() as model:
        # Seasonal effects with zero-sum constraint (NO intercept!)
        seasonal = pm.ZeroSumNormal("seasonal", sigma=5.0, shape=n_periods)
        
        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        
        # Mean is ONLY seasonal
        mu = seasonal[period_index]
        
        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        # Sample
        idata = pm.sample(1000, tune=500, chains=2, random_seed=RANDOM_SEED, progressbar=False)
    
    return {
        "model": model,
        "idata": idata,
        "posterior_seasonal": az.extract(idata.posterior, var_names="seasonal").values,
        "posterior_sigma": az.extract(idata.posterior, var_names="sigma").values,
    }


def model_with_intercept(y: np.ndarray, period_index: np.ndarray, n_periods: int) -> dict[str, Any]:
    """
    Fit a model with intercept + ZeroSumNormal seasonal component.
    This model can fit all components correctly.
    """
    with pm.Model() as model:
        # Intercept (captures the constant/DC component)
        intercept = pm.Normal("intercept", mu=0, sigma=20)
        
        # Seasonal effects with zero-sum constraint
        seasonal = pm.ZeroSumNormal("seasonal", sigma=5.0, shape=n_periods)
        
        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        
        # Mean includes intercept + seasonal
        mu = intercept + seasonal[period_index]
        
        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        # Sample
        idata = pm.sample(1000, tune=500, chains=2, random_seed=RANDOM_SEED, progressbar=False)
    
    return {
        "model": model,
        "idata": idata,
        "posterior_intercept": az.extract(idata.posterior, var_names="intercept").values,
        "posterior_seasonal": az.extract(idata.posterior, var_names="seasonal").values,
        "posterior_sigma": az.extract(idata.posterior, var_names="sigma").values,
    }


def demonstrate_invisible_constant():
    """
    Main demonstration showing that constant signals are 'invisible' 
    to models without intercepts.
    """
    print("\n" + "=" * 70)
    print("PART 3: Empirical Demonstration with PyMC Models")
    print("=" * 70)
    
    # Parameters
    n_periods = 12  # Monthly seasonality
    n_cycles = 4
    true_constant = 10.0
    seasonal_amplitude = 2.0
    noise_std = 0.3
    
    # Create data
    t, y, true_seasonal = create_seasonal_data(
        n_periods=n_periods,
        n_cycles=n_cycles,
        constant=true_constant,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    period_index = t % n_periods
    
    print(f"\nData generated with:")
    print(f"  True constant (intercept): {true_constant}")
    print(f"  Seasonal amplitude: {seasonal_amplitude}")
    print(f"  True seasonal sum: {true_seasonal.sum():.10f} (should be ~0)")
    print(f"  Data mean: {y.mean():.4f}")
    
    # Model 1: WITHOUT intercept
    print("\n" + "-" * 50)
    print("Model 1: ZeroSumNormal ONLY (no intercept)")
    print("-" * 50)
    
    result1 = model_without_intercept(y, period_index, n_periods)
    
    # Compute predictions and residuals
    seasonal_mean_1 = result1["posterior_seasonal"].mean(axis=1)
    predictions_1 = seasonal_mean_1[period_index]
    residuals_1 = y - predictions_1
    
    print(f"\nEstimated seasonal effects (mean): {seasonal_mean_1[:3].round(3)}...")
    print(f"Sum of estimated seasonal: {seasonal_mean_1.sum():.6f}")
    print(f"Predictions mean: {predictions_1.mean():.4f}")
    print(f"Residuals mean: {residuals_1.mean():.4f}")
    print(f"Residuals std: {residuals_1.std():.4f}")
    
    print(f"\n*** The constant {true_constant} is INVISIBLE! ***")
    print(f"*** Residuals capture the unfitted constant + noise ***")
    
    # Model 2: WITH intercept
    print("\n" + "-" * 50)
    print("Model 2: Intercept + ZeroSumNormal")
    print("-" * 50)
    
    result2 = model_with_intercept(y, period_index, n_periods)
    
    intercept_mean = result2["posterior_intercept"].mean()
    seasonal_mean_2 = result2["posterior_seasonal"].mean(axis=1)
    predictions_2 = intercept_mean + seasonal_mean_2[period_index]
    residuals_2 = y - predictions_2
    
    print(f"\nEstimated intercept: {intercept_mean:.4f} (true: {true_constant})")
    print(f"Estimated seasonal effects (mean): {seasonal_mean_2[:3].round(3)}...")
    print(f"True seasonal effects: {true_seasonal[:3].round(3)}...")
    print(f"Sum of estimated seasonal: {seasonal_mean_2.sum():.6f}")
    print(f"Predictions mean: {predictions_2.mean():.4f}")
    print(f"Residuals mean: {residuals_2.mean():.4f}")
    print(f"Residuals std: {residuals_2.std():.4f} (close to noise_std={noise_std})")
    
    print(f"\n*** With intercept, the constant IS captured! ***")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'No Intercept':<20} {'With Intercept':<20}")
    print("-" * 70)
    print(f"{'Residuals mean':<30} {residuals_1.mean():<20.4f} {residuals_2.mean():<20.4f}")
    print(f"{'Residuals std':<30} {residuals_1.std():<20.4f} {residuals_2.std():<20.4f}")
    print(f"{'MSE':<30} {(residuals_1**2).mean():<20.4f} {(residuals_2**2).mean():<20.4f}")
    print(f"{'Constant captured?':<30} {'NO':<20} {'YES':<20}")


def test_pure_constant_signal():
    """
    Test with a PURE constant signal (no seasonality) to show
    it's completely invisible to the no-intercept model.
    """
    print("\n" + "=" * 70)
    print("PART 4: Pure Constant Signal (The Orthogonal Function)")
    print("=" * 70)
    
    n_periods = 12
    n_cycles = 4
    n_total = n_periods * n_cycles
    
    # Pure constant signal + small noise
    constant = 7.5
    y = constant + rng.normal(0, 0.1, n_total)
    period_index = np.arange(n_total) % n_periods
    
    print(f"\nPure constant signal: y = {constant} + noise")
    print(f"This is THE orthogonal function to the zero-sum subspace!")
    
    print("\n" + "-" * 50)
    print("Fitting with ZeroSumNormal ONLY (no intercept)")
    print("-" * 50)
    
    result = model_without_intercept(y, period_index, n_periods)
    seasonal_mean = result["posterior_seasonal"].mean(axis=1)
    predictions = seasonal_mean[period_index]
    residuals = y - predictions
    
    print(f"\nEstimated seasonal effects: all near 0 (as expected)")
    print(f"  First 3 values: {seasonal_mean[:3].round(4)}")
    print(f"  Max absolute: {np.abs(seasonal_mean).max():.4f}")
    print(f"\nPredictions mean: {predictions.mean():.6f} (should be ~0)")
    print(f"Residuals mean: {residuals.mean():.4f} (should be ~{constant})")
    
    print(f"\n*** The ENTIRE constant signal is in the residuals! ***")
    print(f"*** This proves the constant is orthogonal to ZeroSumNormal space ***")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# ZERO-SUM CONSTRAINT ORTHOGONALITY DEMONSTRATION")
    print("#" * 70)
    
    # Part 1: Mathematical verification
    verify_orthogonality()
    
    # Part 2: Fourier analysis
    fourier_analysis()
    
    # Part 3: Empirical demonstration with seasonal data
    demonstrate_invisible_constant()
    
    # Part 4: Pure constant signal test
    test_pure_constant_signal()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The zero-sum constraint on seasonal effects (as implemented by ZeroSumNormal)
creates an (n-1)-dimensional subspace of R^n. The orthogonal complement is
the 1-dimensional space of constant functions.

KEY INSIGHTS:
1. A constant signal is the UNIQUE (up to scaling) function orthogonal to 
   all zero-sum constrained functions.

2. In Fourier terms: zero-sum ⟺ no DC component (k=0 harmonic = 0)

3. If you model y = seasonal (with ZeroSumNormal), constant offsets are 
   INVISIBLE and will appear entirely in the residuals.

4. The CORRECT model is y = intercept + seasonal, where:
   - intercept captures the DC component (overall level)
   - seasonal (with zero-sum) captures all other harmonics
   
5. The zero-sum constraint is for IDENTIFIABILITY, not limitation:
   Without it, intercept and seasonal would be non-identifiable
   (you could add any constant to seasonal and subtract from intercept).

ANSWER TO THE ORIGINAL QUESTION:
The constraint IS sensibly implemented when used with an intercept term.
The "invisible" timeseries (constant function) is only invisible if you
incorrectly omit the intercept from your model specification.
""")
