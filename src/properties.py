"""
Thermophysical properties of common fluids and materials.
"""

import numpy as np
from typing import NamedTuple


class FluidProperties(NamedTuple):
    """Container for fluid properties at a given temperature."""
    T: float       # Temperature (K)
    rho: float     # Density (kg/m³)
    cp: float      # Specific heat (J/kg·K)
    k: float       # Thermal conductivity (W/m·K)
    mu: float      # Dynamic viscosity (Pa·s)
    nu: float      # Kinematic viscosity (m²/s)
    alpha: float   # Thermal diffusivity (m²/s)
    Pr: float      # Prandtl number


def air_properties(T: float) -> FluidProperties:
    """
    Properties of air at atmospheric pressure.

    Parameters
    ----------
    T : float
        Temperature in Kelvin (250-400 K range for accuracy)

    Returns
    -------
    FluidProperties
        Named tuple with all properties

    Examples
    --------
    >>> props = air_properties(300)
    >>> props.Pr
    0.707
    """
    # Polynomial fits valid for 250-400 K
    T_ref = T / 300  # Normalised temperature

    rho = 1.177 / T_ref  # Ideal gas approximation
    cp = 1007  # Nearly constant
    k = 0.0263 * T_ref**0.8
    mu = 1.85e-5 * T_ref**0.7

    nu = mu / rho
    alpha = k / (rho * cp)
    Pr = nu / alpha

    return FluidProperties(T=T, rho=rho, cp=cp, k=k, mu=mu, nu=nu, alpha=alpha, Pr=Pr)


def water_properties(T: float) -> FluidProperties:
    """
    Properties of liquid water at atmospheric pressure.

    Parameters
    ----------
    T : float
        Temperature in Kelvin (280-370 K range for accuracy)

    Returns
    -------
    FluidProperties
        Named tuple with all properties
    """
    # Simplified correlations
    T_C = T - 273.15  # Celsius

    rho = 1000 - 0.05 * (T_C - 20)**2 / 80  # Approximate
    cp = 4180
    k = 0.569 + 0.0019 * T_C
    mu = 0.001 * np.exp(-0.02 * T_C)  # Approximate

    nu = mu / rho
    alpha = k / (rho * cp)
    Pr = nu / alpha

    return FluidProperties(T=T, rho=rho, cp=cp, k=k, mu=mu, nu=nu, alpha=alpha, Pr=Pr)


# Material thermal conductivities (W/m·K) at ~300K
CONDUCTIVITY = {
    'copper': 401,
    'aluminium': 237,
    'steel_carbon': 60,
    'steel_stainless': 15,
    'glass': 1.4,
    'concrete': 1.0,
    'wood_oak': 0.17,
    'fiberglass': 0.04,
    'air': 0.026,
}


def get_conductivity(material: str) -> float:
    """
    Get thermal conductivity of a material.

    Parameters
    ----------
    material : str
        Material name (case insensitive)

    Returns
    -------
    float
        Thermal conductivity (W/m·K)
    """
    key = material.lower().replace(' ', '_')
    if key not in CONDUCTIVITY:
        available = ', '.join(CONDUCTIVITY.keys())
        raise ValueError(f"Unknown material '{material}'. Available: {available}")
    return CONDUCTIVITY[key]
