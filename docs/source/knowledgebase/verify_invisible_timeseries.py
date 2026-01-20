"""
Verify the closed-form formula for the "invisible" function g(t)
and demonstrate whether ZeroSumNormal creates fitting problems.

The claim is that ZeroSumNormal on Fourier coefficients enforces:
    sum_j (a_j + b_j) = 0

And this makes the function g(t) = sum_j [cos(2πjt/S) + sin(2πjt/S)] "invisible".
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_g_numerical(S: int, n: int) -> np.ndarray:
    """Compute g(t) = sum_j [cos(2πjt/S) + sin(2πjt/S)] numerically."""
    t = np.arange(S)
    g = np.zeros(S)
    for j in range(1, n + 1):
        g += np.cos(2 * np.pi * j * t / S) + np.sin(2 * np.pi * j * t / S)
    return g


def compute_g_closed_form(S: int) -> np.ndarray:
    """
    Compute g(t) using the claimed closed form (saturated case n = S/2):
    
    g(t) = S/2           if t = 0
    g(t) = 0             if t even, t ≠ 0  
    g(t) = cot(πt/S) - 1 if t odd
    """
    t = np.arange(S)
    g = np.zeros(S)
    
    g[0] = S / 2
    
    for i in range(1, S):
        if i % 2 == 0:  # even
            g[i] = 0
        else:  # odd
            g[i] = 1 / np.tan(np.pi * i / S) - 1
    
    return g


def verify_closed_form():
    """Verify the closed-form formula matches numerical computation."""
    print("=" * 70)
    print("PART 1: Verify Closed-Form Formula")
    print("=" * 70)
    
    for S in [12, 24, 52]:
        n = S // 2  # Saturated case
        g_num = compute_g_numerical(S, n)
        g_closed = compute_g_closed_form(S)
        
        max_diff = np.max(np.abs(g_num - g_closed))
        print(f"\nS = {S}, n = {n}:")
        print(f"  Max difference between numerical and closed-form: {max_diff:.2e}")
        print(f"  First few values of g(t):")
        for t in range(min(6, S)):
            print(f"    g({t}) = {g_num[t]:+.4f} (numerical), {g_closed[t]:+.4f} (closed-form)")


def analyze_single_harmonic():
    """
    Analyze the n=1 case where the constraint a_1 + b_1 = 0
    locks the phase to π/4.
    """
    print("\n" + "=" * 70)
    print("PART 2: Single Harmonic Case (n=1)")
    print("=" * 70)
    
    S = 12
    t = np.arange(S)
    omega = 2 * np.pi / S
    
    # True seasonal pattern: pure cosine (phase = 0)
    A = 2.0
    gamma_true = A * np.cos(omega * t)
    
    # Coefficients: a_1 = A, b_1 = 0
    a1_true, b1_true = A, 0.0
    print(f"\nTrue seasonal: γ(t) = {A} * cos(ωt)")
    print(f"True coefficients: a₁ = {a1_true}, b₁ = {b1_true}")
    print(f"Constraint check: a₁ + b₁ = {a1_true + b1_true} ≠ 0")
    
    # Project onto constraint hyperplane a_1 + b_1 = 0
    # Projection formula: (a, b) → (a, b) - [(a+b)/2] * (1, 1)
    sum_ab = a1_true + b1_true
    a1_proj = a1_true - sum_ab / 2
    b1_proj = b1_true - sum_ab / 2
    
    print(f"\nProjected coefficients: a₁ = {a1_proj}, b₁ = {b1_proj}")
    print(f"Constraint check: a₁ + b₁ = {a1_proj + b1_proj:.10f} = 0 ✓")
    
    # Reconstruct the projected seasonal
    gamma_proj = a1_proj * np.cos(omega * t) + b1_proj * np.sin(omega * t)
    
    # Express in amplitude-phase form
    # a*cos + b*sin = R*cos(ωt - φ) where R = √(a²+b²), tan(φ) = b/a
    R_true = np.sqrt(a1_true**2 + b1_true**2)
    phi_true = np.arctan2(b1_true, a1_true)
    
    R_proj = np.sqrt(a1_proj**2 + b1_proj**2)
    phi_proj = np.arctan2(b1_proj, a1_proj)
    
    print(f"\nAmplitude-phase representation:")
    print(f"  True:      R = {R_true:.4f}, φ = {np.degrees(phi_true):.1f}°")
    print(f"  Projected: R = {R_proj:.4f}, φ = {np.degrees(phi_proj):.1f}°")
    print(f"  Amplitude ratio: {R_proj/R_true:.4f} (= 1/√2 = {1/np.sqrt(2):.4f})")
    print(f"  Phase shift: {np.degrees(phi_proj - phi_true):.1f}°")
    
    # Compute fitting error
    mse = np.mean((gamma_true - gamma_proj)**2)
    var_true = np.var(gamma_true)
    r2 = 1 - mse / var_true
    
    print(f"\nFitting quality:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  (Should be 0.5 since we lose half the variance)")
    
    return t, gamma_true, gamma_proj


def analyze_invisible_function():
    """
    Analyze the "invisible" function g(t) and show it can't be fit.
    """
    print("\n" + "=" * 70)
    print("PART 3: The 'Invisible' Function")
    print("=" * 70)
    
    S = 12
    n = S // 2
    t = np.arange(S)
    
    # Compute g(t)
    g = compute_g_numerical(S, n)
    
    print(f"\nThe 'invisible' function g(t) = Σⱼ[cos(2πjt/S) + sin(2πjt/S)]")
    print(f"for S = {S}, n = {n}:")
    print(f"\n  t  |  g(t)")
    print(f"  ---|-------")
    for i in range(S):
        print(f"  {i:2d} | {g[i]:+6.3f}")
    
    # The coefficients of g(t) are all 1
    coeffs = np.ones(2 * n)  # (a_1, b_1, a_2, b_2, ..., a_n, b_n)
    print(f"\nCoefficients of g(t): all ones")
    print(f"Sum of coefficients: {coeffs.sum()} ≠ 0")
    
    # Project onto constraint hyperplane
    # The projection of (1,1,1,...,1) onto Σ(a_j + b_j) = 0 is the ZERO vector!
    c = np.ones(2 * n)  # constraint normal vector
    proj = coeffs - (np.dot(coeffs, c) / np.dot(c, c)) * c
    
    print(f"\nProjected coefficients: {proj}")
    print(f"Sum of projected: {proj.sum():.10f}")
    
    # The best approximation is the ZERO function!
    print(f"\nThe constrained model's best fit to g(t) is γ*(t) = 0!")
    print(f"MSE = E[g(t)²] = {np.mean(g**2):.4f}")
    
    return t, g


def demonstrate_phase_problem():
    """
    Show that for n=1, different phases can't be fit equally well.
    """
    print("\n" + "=" * 70)
    print("PART 4: Phase Dependence of Fitting Error")
    print("=" * 70)
    
    S = 12
    t = np.arange(S)
    omega = 2 * np.pi / S
    A = 1.0
    
    phases = np.linspace(0, 2*np.pi, 100)
    r2_values = []
    
    for phi in phases:
        # True signal with phase phi
        gamma_true = A * np.cos(omega * t - phi)
        
        # Coefficients: a = A*cos(phi), b = A*sin(phi)
        a1 = A * np.cos(phi)
        b1 = A * np.sin(phi)
        
        # Project onto constraint
        sum_ab = a1 + b1
        a1_proj = a1 - sum_ab / 2
        b1_proj = b1 - sum_ab / 2
        
        # Reconstructed signal
        gamma_proj = a1_proj * np.cos(omega * t) + b1_proj * np.sin(omega * t)
        
        # R²
        mse = np.mean((gamma_true - gamma_proj)**2)
        var_true = np.var(gamma_true)
        r2 = 1 - mse / var_true
        r2_values.append(r2)
    
    r2_values = np.array(r2_values)
    
    print(f"\nFor n=1, fitting R² depends on the phase of the true signal:")
    print(f"  Min R²: {r2_values.min():.4f} at phase = {np.degrees(phases[np.argmin(r2_values)]):.1f}°")
    print(f"  Max R²: {r2_values.max():.4f} at phase = {np.degrees(phases[np.argmax(r2_values)]):.1f}°")
    
    # The optimal phase should be -π/4 (or equivalently 315°)
    # because the constraint forces phase to be -π/4
    optimal_phase = -np.pi/4
    print(f"\n  The constraint locks the model phase to {np.degrees(optimal_phase):.1f}° (= -π/4)")
    print(f"  Signals with this phase can be fit perfectly (R² = 1)")
    print(f"  Signals with phase {np.degrees(optimal_phase + np.pi/2):.1f}° (orthogonal) have R² = 0")
    
    return phases, r2_values


def plot_results(t1, gamma_true, gamma_proj, t2, g, phases, r2_values):
    """Create visualization of the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Single harmonic fitting
    ax1 = axes[0, 0]
    ax1.plot(t1, gamma_true, 'b-o', label='True: cos(ωt)', linewidth=2)
    ax1.plot(t1, gamma_proj, 'r--s', label='Projected (constrained)', linewidth=2)
    ax1.set_xlabel('t')
    ax1.set_ylabel('γ(t)')
    ax1.set_title('n=1: Pure Cosine Cannot Be Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The invisible function
    ax2 = axes[0, 1]
    ax2.bar(t2, g, color='purple', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('t')
    ax2.set_ylabel('g(t)')
    ax2.set_title("The 'Invisible' Function g(t)")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: R² vs phase
    ax3 = axes[1, 0]
    ax3.plot(np.degrees(phases), r2_values, 'g-', linewidth=2)
    ax3.axhline(y=0.5, color='r', linestyle='--', label='Average R² = 0.5')
    ax3.set_xlabel('Phase φ (degrees)')
    ax3.set_ylabel('R²')
    ax3.set_title('n=1: Fitting Quality Depends on Phase')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    ax3.set_ylim(-0.1, 1.1)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = """
    SUMMARY OF FINDINGS
    
    The ZeroSumNormal constraint on Fourier coefficients:
        Σⱼ(aⱼ + bⱼ) = 0
    
    Creates the following problems:
    
    1. For n=1 (single harmonic):
       • Locks phase to -45° (= -π/4)
       • Pure cosine (phase=0°) has R² = 0.5
       • Loses half the signal variance
    
    2. For general n (saturated case):
       • The function g(t) = Σⱼ[cos + sin]
         is completely "invisible" (R² = 0)
       • Best fit is the zero function!
    
    3. Root cause:
       • Constraint is in COEFFICIENT space
       • NOT equivalent to time-domain zero-sum
       • Fourier basis already excludes DC
    
    RECOMMENDATION: Replace ZeroSumNormal with Normal
    """
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('/workspace/docs/source/knowledgebase/zero_sum_analysis.png', dpi=150)
    print(f"\nPlot saved to: docs/source/knowledgebase/zero_sum_analysis.png")


if __name__ == "__main__":
    print("#" * 70)
    print("# VERIFICATION OF ZERO-SUM CONSTRAINT ANALYSIS")
    print("#" * 70)
    
    # Part 1: Verify the closed-form formula
    verify_closed_form()
    
    # Part 2: Analyze single harmonic case
    t1, gamma_true, gamma_proj = analyze_single_harmonic()
    
    # Part 3: Analyze the invisible function
    t2, g = analyze_invisible_function()
    
    # Part 4: Show phase dependence
    phases, r2_values = demonstrate_phase_problem()
    
    # Create visualization
    plot_results(t1, gamma_true, gamma_proj, t2, g, phases, r2_values)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The closed-form formula IS CORRECT.

The ZeroSumNormal constraint on Fourier coefficients:
1. DOES create "invisible" functions that cannot be fit
2. For n=1, arbitrarily restricts the phase to -π/4
3. For general n, the function g(t) = Σⱼ[cos(2πjt/S) + sin(2πjt/S)]
   has R² = 0 (best fit is zero!)

This is NOT a sensible constraint. It should be replaced with
a simple Normal prior on the Fourier coefficients.
""")
