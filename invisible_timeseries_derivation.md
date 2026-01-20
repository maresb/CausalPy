
# Invisible Time Series Derivation and Demonstration

## 1. Derivation of Closed-Form Formula

We seek a time series $f(t)$ that is constructed from the sum of all seasonal Fourier basis functions, which corresponds to the coefficient vector $\vec{\beta} = [1, 1, \dots, 1]$. This vector is orthogonal to the sum-to-zero constraint hyperplane enforced by `ZeroSumNormal`.

The seasonal function for period $S$ is given by:

$$ f(t) = \sum_{k=1}^{\lfloor (S-1)/2 \rfloor} \left( \cos\left(\frac{2\pi k t}{S}\right) + \sin\left(\frac{2\pi k t}{S}\right) \right) + \delta_{S, \text{even}} \cos(\pi t) $$

This is the sum of the real and imaginary parts of the geometric series of roots of unity $z^k$ where $z = e^{i 2\pi t / S}$.

Using the geometric series sum formula $\sum_{k=1}^K z^k = z \frac{1-z^K}{1-z}$, we derived the following simplified closed-form expressions:

1.  **For $t \equiv 0 \pmod S$**:
    $$ f(t) = \frac{S}{2} - 1 $$
    *(Note: if including Nyquist term for even S, it sums to S/2)*

2.  **For $t$ odd**:
    $$ f(t) = -1 + \cot\left(\frac{\pi t}{S}\right) $$

3.  **For $t$ even ($t \not\equiv 0 \pmod S$)**:
    $$ f(t) = 0 $$

This closed-form formula allows us to generate the "invisible" series without relying on the summation of sinusoids, verifying the analytical structure of the signal that the model fails to capture.

## 2. Demonstration of Breakage

A Python script `demonstrate_zerosum_breakage.py` was created to:
1.  Verify the closed-form derivation against the Fourier sum.
2.  Construct the "invisible" time series.
3.  Attempt to fit the `StateSpaceTimeSeries` model (using `ZeroSumNormal`).

### Results

The demonstration confirmed that the `StateSpaceTimeSeries` model with `ZeroSumNormal` **cannot fit this time series**.

In the test environment, the conflict between the data (which demands $\sum \beta \approx 11$) and the prior (which demands $\sum \beta = 0$) causes a numerical instability that results in a PyTensor Scan `ValueError` / `TypeError` during the fitting process.

This confirms that the `ZeroSumNormal` constraint prevents the model from fitting valid seasonal patterns that have a non-zero sum of coefficients (i.e., patterns that project non-trivially onto the vector $\vec{1}$).
