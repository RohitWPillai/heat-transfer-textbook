"""
Conduction heat transfer calculations.
"""

import numpy as np


# =============================================================================
# Thermal Resistance
# =============================================================================

def resistance_conduction_plane(L: float, k: float, A: float) -> float:
    """Thermal resistance for conduction through a plane wall."""
    return L / (k * A)


def resistance_conduction_cylinder(r1: float, r2: float, k: float, L: float) -> float:
    """Thermal resistance for radial conduction through a cylinder."""
    return np.log(r2 / r1) / (2 * np.pi * k * L)


def resistance_conduction_sphere(r1: float, r2: float, k: float) -> float:
    """Thermal resistance for radial conduction through a sphere."""
    return (1/r1 - 1/r2) / (4 * np.pi * k)


def resistance_convection(h: float, A: float) -> float:
    """Thermal resistance for convection."""
    return 1 / (h * A)


# =============================================================================
# Fins
# =============================================================================

def fin_parameter_m(h: float, P: float, k: float, Ac: float) -> float:
    """Calculate fin parameter m = sqrt(hP/kAc)."""
    return np.sqrt(h * P / (k * Ac))


def fin_heat_transfer_adiabatic(m: float, L: float, theta_b: float,
                                 h: float, P: float, k: float, Ac: float) -> float:
    """
    Heat transfer from a fin with adiabatic tip.

    Parameters
    ----------
    m : float
        Fin parameter sqrt(hP/kAc)
    L : float
        Fin length (m)
    theta_b : float
        Excess temperature at base (T_b - T_inf)
    h, P, k, Ac : float
        Fin parameters

    Returns
    -------
    float
        Heat transfer rate (W)
    """
    return np.sqrt(h * P * k * Ac) * theta_b * np.tanh(m * L)


def fin_efficiency_adiabatic(m: float, L: float) -> float:
    """Fin efficiency for adiabatic tip: η = tanh(mL)/(mL)."""
    mL = m * L
    if mL < 1e-6:
        return 1.0
    return np.tanh(mL) / mL


def fin_effectiveness(Q_fin: float, h: float, Ac_base: float, theta_b: float) -> float:
    """Fin effectiveness: ε = Q_fin / Q_without_fin."""
    Q_no_fin = h * Ac_base * theta_b
    return Q_fin / Q_no_fin


# =============================================================================
# Transient Conduction
# =============================================================================

def biot_number(h: float, Lc: float, k: float) -> float:
    """Biot number Bi = hLc/k."""
    return h * Lc / k


def lumped_capacitance_valid(Bi: float, threshold: float = 0.1) -> bool:
    """Check if lumped capacitance is valid (Bi < threshold)."""
    return Bi < threshold


def time_constant(rho: float, V: float, cp: float, h: float, As: float) -> float:
    """Time constant τ = ρVcp/(hAs)."""
    return rho * V * cp / (h * As)


def lumped_temperature(t: float, Ti: float, Tinf: float, tau: float) -> float:
    """Temperature using lumped capacitance method."""
    return Tinf + (Ti - Tinf) * np.exp(-t / tau)


def fourier_number(alpha: float, t: float, L: float) -> float:
    """Fourier number Fo = αt/L²."""
    return alpha * t / L**2


# =============================================================================
# Numerical Methods
# =============================================================================

def solve_1d_steady(T_left: float, T_right: float, N: int) -> np.ndarray:
    """
    Solve 1D steady conduction with Dirichlet BCs.

    Returns linear temperature profile.
    """
    return np.linspace(T_left, T_right, N)


def explicit_stability_limit(dx: float, alpha: float) -> float:
    """Maximum stable time step for explicit 1D conduction."""
    return dx**2 / (2 * alpha)
