"""
Heat transfer correlations for convection.
"""

import numpy as np


# =============================================================================
# External Flow
# =============================================================================

def nusselt_flat_plate(Re: float, Pr: float, turbulent: bool = False) -> float:
    """
    Average Nusselt number for flow over a flat plate.

    Parameters
    ----------
    Re : float
        Reynolds number based on plate length
    Pr : float
        Prandtl number
    turbulent : bool
        If True, use turbulent correlation

    Returns
    -------
    float
        Average Nusselt number

    Notes
    -----
    Laminar (Re < 5×10⁵): Nu = 0.664 Re^0.5 Pr^(1/3)
    Turbulent: Nu = 0.037 Re^0.8 Pr^(1/3)
    """
    if turbulent:
        return 0.037 * Re**0.8 * Pr**(1/3)
    else:
        return 0.664 * Re**0.5 * Pr**(1/3)


def nusselt_cylinder_crossflow(Re: float, Pr: float) -> float:
    """
    Average Nusselt number for cross-flow over a cylinder.

    Churchill-Bernstein correlation, valid for Re·Pr > 0.2.

    Parameters
    ----------
    Re : float
        Reynolds number based on diameter
    Pr : float
        Prandtl number

    Returns
    -------
    float
        Average Nusselt number
    """
    term1 = 0.62 * Re**0.5 * Pr**(1/3)
    term2 = (1 + (0.4/Pr)**(2/3))**0.25
    term3 = (1 + (Re/282000)**(5/8))**(4/5)

    return 0.3 + (term1 / term2) * term3


def nusselt_sphere(Re: float, Pr: float, mu_ratio: float = 1.0) -> float:
    """
    Average Nusselt number for flow over a sphere.

    Whitaker correlation.

    Parameters
    ----------
    Re : float
        Reynolds number based on diameter
    Pr : float
        Prandtl number
    mu_ratio : float
        Ratio μ_inf/μ_s (viscosity at freestream/surface temperature)

    Returns
    -------
    float
        Average Nusselt number
    """
    return 2 + (0.4 * Re**0.5 + 0.06 * Re**(2/3)) * Pr**0.4 * mu_ratio**0.25


# =============================================================================
# Internal Flow
# =============================================================================

def nusselt_pipe_laminar(constant_T: bool = True) -> float:
    """
    Nusselt number for fully developed laminar pipe flow.

    Parameters
    ----------
    constant_T : bool
        True for constant wall temperature, False for constant heat flux

    Returns
    -------
    float
        Nusselt number
    """
    return 3.66 if constant_T else 4.36


def nusselt_pipe_turbulent(Re: float, Pr: float, heating: bool = True) -> float:
    """
    Nusselt number for fully developed turbulent pipe flow.

    Dittus-Boelter correlation.

    Parameters
    ----------
    Re : float
        Reynolds number based on diameter
    Pr : float
        Prandtl number
    heating : bool
        True if fluid is being heated, False if cooled

    Returns
    -------
    float
        Nusselt number

    Notes
    -----
    Valid for: 0.6 ≤ Pr ≤ 160, Re > 10000, L/D > 10
    """
    n = 0.4 if heating else 0.3
    return 0.023 * Re**0.8 * Pr**n


# =============================================================================
# Natural Convection
# =============================================================================

def nusselt_vertical_plate_natural(Ra: float) -> float:
    """
    Nusselt number for natural convection on a vertical plate.

    Churchill-Chu correlation.

    Parameters
    ----------
    Ra : float
        Rayleigh number

    Returns
    -------
    float
        Average Nusselt number
    """
    if Ra < 1e9:
        # Laminar
        return 0.68 + 0.67 * Ra**0.25 / (1 + (0.492/0.71)**(9/16))**(4/9)
    else:
        # Turbulent
        term = 0.387 * Ra**(1/6) / (1 + (0.492/0.71)**(9/16))**(8/27)
        return (0.825 + term)**2


# =============================================================================
# Dimensionless Numbers
# =============================================================================

def reynolds_number(V: float, L: float, nu: float) -> float:
    """Reynolds number Re = VL/ν."""
    return V * L / nu


def prandtl_number(nu: float, alpha: float) -> float:
    """Prandtl number Pr = ν/α."""
    return nu / alpha


def nusselt_to_h(Nu: float, L: float, k: float) -> float:
    """Convert Nusselt number to heat transfer coefficient."""
    return Nu * k / L


def grashof_number(g: float, beta: float, dT: float, L: float, nu: float) -> float:
    """Grashof number Gr = gβΔTL³/ν²."""
    return g * beta * abs(dT) * L**3 / nu**2


def rayleigh_number(Gr: float, Pr: float) -> float:
    """Rayleigh number Ra = Gr·Pr."""
    return Gr * Pr
