"""
Radiation heat transfer calculations.
"""

import numpy as np

# Physical constants
SIGMA = 5.67e-8  # Stefan-Boltzmann constant (W/m²·K⁴)
C1 = 3.742e8     # First radiation constant (W·μm⁴/m²)
C2 = 1.439e4     # Second radiation constant (μm·K)


# =============================================================================
# Blackbody Radiation
# =============================================================================

def planck(wavelength: float, T: float) -> float:
    """
    Planck's law: spectral blackbody emissive power.

    Parameters
    ----------
    wavelength : float
        Wavelength in micrometers (μm)
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Spectral emissive power (W/m²·μm)
    """
    return C1 / (wavelength**5 * (np.exp(C2 / (wavelength * T)) - 1))


def stefan_boltzmann(T: float) -> float:
    """Total blackbody emissive power Eb = σT⁴."""
    return SIGMA * T**4


def wien_peak_wavelength(T: float) -> float:
    """Peak wavelength from Wien's displacement law (μm)."""
    return 2898 / T


# =============================================================================
# Radiative Properties
# =============================================================================

def emissive_power(eps: float, T: float) -> float:
    """Emissive power of a gray surface E = εσT⁴."""
    return eps * SIGMA * T**4


def radiation_heat_transfer_coefficient(eps: float, Ts: float, Tinf: float) -> float:
    """
    Linearised radiation heat transfer coefficient.

    hr = εσ(Ts + Tinf)(Ts² + Tinf²)
    """
    return eps * SIGMA * (Ts + Tinf) * (Ts**2 + Tinf**2)


# =============================================================================
# View Factors
# =============================================================================

def view_factor_parallel_rectangles(a: float, b: float, c: float) -> float:
    """
    View factor between two parallel aligned rectangles.

    Parameters
    ----------
    a : float
        Width of rectangles
    b : float
        Height of rectangles
    c : float
        Distance between rectangles

    Returns
    -------
    float
        View factor F_12
    """
    X = a / c
    Y = b / c

    term1 = np.log((1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2))
    term2 = X * np.sqrt(1 + Y**2) * np.arctan(X / np.sqrt(1 + Y**2))
    term3 = Y * np.sqrt(1 + X**2) * np.arctan(Y / np.sqrt(1 + X**2))
    term4 = -X * np.arctan(X) - Y * np.arctan(Y)

    return (2 / (np.pi * X * Y)) * (term1/2 + term2 + term3 + term4)


def view_factor_coaxial_disks(r1: float, r2: float, h: float) -> float:
    """
    View factor between two coaxial parallel disks.

    Parameters
    ----------
    r1 : float
        Radius of disk 1
    r2 : float
        Radius of disk 2
    h : float
        Distance between disks

    Returns
    -------
    float
        View factor F_12
    """
    R1 = r1 / h
    R2 = r2 / h
    S = 1 + (1 + R2**2) / R1**2

    return 0.5 * (S - np.sqrt(S**2 - 4 * (R2/R1)**2))


# =============================================================================
# Radiation Exchange
# =============================================================================

def radiation_exchange_blackbody(A1: float, F12: float, T1: float, T2: float) -> float:
    """
    Heat transfer between two blackbody surfaces.

    Q = A1·F12·σ(T1⁴ - T2⁴)
    """
    return A1 * F12 * SIGMA * (T1**4 - T2**4)


def radiation_exchange_gray_two_surface(T1: float, T2: float,
                                         eps1: float, eps2: float,
                                         A1: float, A2: float,
                                         F12: float) -> float:
    """
    Heat transfer between two gray surfaces in an enclosure.

    Uses resistance network method.
    """
    R1 = (1 - eps1) / (eps1 * A1)      # Surface 1 resistance
    R12 = 1 / (A1 * F12)                # Space resistance
    R2 = (1 - eps2) / (eps2 * A2)      # Surface 2 resistance

    R_total = R1 + R12 + R2

    return SIGMA * (T1**4 - T2**4) / R_total


def radiation_shield_factor(N: int) -> float:
    """
    Reduction factor for N radiation shields.

    Q_with_shields = Q_no_shields / (N + 1)
    """
    return 1 / (N + 1)
