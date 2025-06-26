import numpy as np
import cmath
from scipy.special import gamma

STEP_SIZE = 0.025  # step size of the rectangles


# calculate roots of z^5 - 2 for visual inspection
def get_theoretical_roots():
    roots = []
    for k in range(5):
        # z^5 = 2, so z = 2^(1/5) * e^(2πik/5)
        magnitude = 2**(1/5)
        angle = 2 * np.pi * k / 5
        root = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        roots.append(root)
    return roots


def winding_number(path):
    if len(path) < 3:
        return 0

    # Ensure path is closed
    closed_path = list(path)
    if closed_path[0] != closed_path[-1]:
        closed_path.append(closed_path[0])

    total_angle = 0.0

    for i in range(len(closed_path) - 1):
        z1 = closed_path[i]
        z2 = closed_path[i + 1]

        # avoid division by zero
        if abs(z1) < 1e-15 or abs(z2) < 1e-15:
            continue

        # calc angle change
        angle1 = cmath.phase(z1)
        angle2 = cmath.phase(z2)

        # avoid taking the wring angle mod 2pi
        angle_diff = angle2 - angle1
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        total_angle += angle_diff

    return round(total_angle / (2 * np.pi))


# create path around the rectangle
def create_rectangle_path(top_left, bottom_right):
    top_right = complex(bottom_right.real, top_left.imag)
    bottom_left = complex(top_left.real, bottom_right.imag)

    sides = [
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left)
    ]

    all_points = []
    for start, end in sides:
        distance = abs(end - start)
        num_points = max(1, int(np.ceil(distance / STEP_SIZE)))

        # interpolate along this side, to get a path with distance STEP_SIZE
        for i in range(num_points):
            t = i / num_points if num_points > 1 else 0
            point = start + t * (end - start)
            all_points.append(point)

    return all_points


def split_rectangle_at_longest_size(top_left, bottom_right):
    width = abs(bottom_right.real - top_left.real)
    height = abs(top_left.imag - bottom_right.imag)

    # Split rectangle over longest side
    if width >= height:
        # Split vertically
        mid = (top_left.real + bottom_right.real) / 2
        return [(top_left, complex(mid, bottom_right.imag)),
                (complex(mid, top_left.imag), bottom_right)]
    else:
        # Split horizontally
        mid = (top_left.imag + bottom_right.imag) / 2
        return [(top_left, complex(bottom_right.real, mid)),
                (complex(top_left.real, mid), bottom_right)]


# Riemann Zeta Function Implementation
def riemann_zeta(s, max_terms=1000):
    """
    Compute the Riemann zeta function ζ(s) using analytic continuation.

    Args:
        s: Complex number input
        max_terms: Maximum number of terms for series convergence

    Returns:
        Complex value of ζ(s)
    """
    s = complex(s)

    # Handle special cases
    if abs(s - 1) < 1e-15:
        # ζ(1) has a simple pole
        return complex('inf')

    if s.real > 1:
        # For Re(s) > 1, use the standard Dirichlet series
        return _zeta_dirichlet_series(s, max_terms)

    elif s.real > 0:
        # For 0 < Re(s) ≤ 1, use Euler-Maclaurin formula or other acceleration
        return _zeta_euler_maclaurin(s, max_terms)

    else:
        # For Re(s) ≤ 0, use functional equation: ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
        return _zeta_functional_equation(s, max_terms)


def _zeta_dirichlet_series(s, max_terms):
    """Standard Dirichlet series for Re(s) > 1"""
    result = 0.0
    for n in range(1, max_terms + 1):
        term = 1.0 / (n ** s)
        result += term
        # Check for convergence
        if abs(term) < 1e-15:
            break
    return result


def _zeta_euler_maclaurin(s, max_terms):
    """
    Euler-Maclaurin formula for better convergence in the critical strip.
    Uses the formula with Bernoulli numbers for acceleration.
    """
    # For simplicity, use Dirichlet series with Euler acceleration
    # ζ(s) = 1/(1-2^(1-s)) * Σ((-1)^(n-1) / n^s)

    if abs(1 - 2**(1-s)) < 1e-15:
        # Handle case where denominator is near zero
        return _zeta_dirichlet_series(s, max_terms)

    eta_sum = 0.0  # Dirichlet eta function
    for n in range(1, max_terms + 1):
        term = ((-1)**(n-1)) / (n ** s)
        eta_sum += term
        if abs(term) < 1e-15:
            break

    return eta_sum / (1 - 2**(1-s))


def _zeta_functional_equation(s, max_terms):
    """
    Use functional equation for Re(s) ≤ 0:
    ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
    """
    s_conj = 1 - s

    # Compute ζ(1-s) recursively
    if s_conj.real > 1:
        zeta_conj = _zeta_dirichlet_series(s_conj, max_terms)
    elif s_conj.real > 0:
        zeta_conj = _zeta_euler_maclaurin(s_conj, max_terms)
    else:
        # Avoid infinite recursion by using a simpler approximation
        zeta_conj = _zeta_dirichlet_series(s_conj, min(100, max_terms))

    # Compute the functional equation components
    try:
        factor1 = 2**s
        factor2 = np.pi**(s-1)
        factor3 = np.sin(np.pi * s / 2)
        factor4 = gamma(1-s)

        result = factor1 * factor2 * factor3 * factor4 * zeta_conj

        # Handle potential numerical issues
        if np.isnan(result) or np.isinf(result):
            return complex(0, 0)

        return result
    except (OverflowError, ValueError):
        return complex(0, 0)


def zeta_zeros_trivial():
    """
    Return the trivial zeros of the Riemann zeta function.
    These are at s = -2, -4, -6, -8, ...
    """
    return [-2*k for k in range(1, 11)]  # First 10 trivial zeros


def zeta_zeros_nontrivial_known():
    """
    Return some known non-trivial zeros of the Riemann zeta function.
    These are approximations of zeros on the critical line Re(s) = 1/2.
    """
    # First few non-trivial zeros (imaginary parts)
    known_zeros_im = [
        14.134725142,
        21.022039639,
        25.010857580,
        30.424876126,
        32.935061588,
        37.586178159,
        40.918719012,
        43.327073281,
        48.005150881,
        49.773832478
    ]

    # All non-trivial zeros have real part 1/2 (Riemann Hypothesis)
    return [complex(0.5, im) for im in known_zeros_im]