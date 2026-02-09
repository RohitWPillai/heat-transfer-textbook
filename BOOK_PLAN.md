# Heat Transfer Textbook - Structure Plan

## Overview

An open-source, interactive heat transfer textbook with Python code throughout.
Target: JOSE publication, general online audience (undergrad level).

## Four-Part Structure

### Part I: Introduction
*Foundations and the big picture*

1. **What is Heat Transfer?**
   - Three modes overview (conduction, convection, radiation)
   - When each dominates
   - Real-world examples (cooking, climate, electronics cooling)

2. **Thermodynamic Foundations**
   - Energy conservation (First Law)
   - Temperature and heat flux
   - Thermal equilibrium

3. **Getting Started with Python**
   - Setting up the environment
   - NumPy, SciPy, Matplotlib basics
   - First example: Newton's law of cooling (ODE solution)

4. **How to Solve a Heat Transfer Problem**
   - Problem-solving as classification (mode, steady/transient, dimensions, BCs)
   - The systematic framework: read, categorise, sketch, assume, balance, check, solve
   - Worked example: cooling of an aluminum sphere (convection + radiation)
   - Bridge to the rest of the book (what pieces are still missing)

**Python focus**: Environment setup, simple ODE solver, first interactive plot

---

### Part II: Convection
*Heat transfer with fluid motion*

4. **Boundary Layer Fundamentals**
   - Velocity and thermal boundary layers
   - Laminar vs turbulent
   - Governing equations (continuity, momentum, energy)

5. **Dimensionless Numbers**
   - Reynolds, Prandtl, Nusselt, Grashof, Rayleigh
   - Physical meaning of each
   - How correlations work

6. **External Forced Convection**
   - Flat plate (laminar and turbulent)
   - Cylinders and spheres
   - Flow over tube banks

7. **Internal Forced Convection**
   - Pipe flow fundamentals
   - Entrance region vs fully developed
   - Constant wall temperature vs constant heat flux

8. **Natural Convection**
   - Buoyancy-driven flow
   - Vertical and horizontal surfaces
   - Enclosures

9. **Heat Exchangers**
   - Types and configurations
   - LMTD method
   - Effectiveness-NTU method
   - Sizing and rating problems

**Python focus**:
- Boundary layer thickness calculator
- Nu correlation library (flat plate, cylinder, sphere, pipe)
- Interactive Re/Pr/Nu explorer
- Heat exchanger design tool (LMTD and ε-NTU)
- Property lookup functions (air, water)

---

### Part III: Conduction
*Heat transfer through solids*

10. **Fourier's Law and the Heat Equation**
    - Derivation of the diffusion equation
    - Thermal conductivity
    - Boundary conditions

11. **Steady-State Conduction**
    - 1D conduction through walls
    - Thermal resistance networks
    - Composite walls and contact resistance

12. **Extended Surfaces (Fins)**
    - Fin equation derivation
    - Fin efficiency and effectiveness
    - Fin arrays and practical design

13. **Transient Conduction**
    - Lumped capacitance method (when valid?)
    - Analytical solutions (Heisler charts)
    - Numerical methods (explicit/implicit schemes)

14. **Multi-dimensional Conduction**
    - 2D steady conduction
    - Shape factors
    - Numerical solution (finite differences)

**Python focus**:
- Thermal resistance calculator
- Fin efficiency solver
- Biot number checker (lumped capacitance validity)
- 2D conduction heatmap solver
- Transient conduction animation

---

### Part IV: Radiation
*Heat transfer by electromagnetic waves*

15. **Fundamentals of Thermal Radiation**
    - Electromagnetic spectrum
    - Blackbody radiation (Planck, Wien, Stefan-Boltzmann)
    - Real surfaces: emissivity, absorptivity, reflectivity

16. **Radiative Properties**
    - Kirchhoff's law
    - Spectral and directional properties
    - Gray body approximation

17. **View Factors**
    - Definition and properties
    - Reciprocity and summation rules
    - Common geometries

18. **Radiation Exchange**
    - Exchange between blackbodies
    - Radiosity method for gray surfaces
    - Radiation shields

19. **Combined Modes**
    - Radiation with convection
    - Radiation in enclosures with convection
    - Practical examples (thermos flask, solar collector)

**Python focus**:
- Blackbody spectrum plotter (interactive wavelength/temperature)
- View factor calculator (common geometries)
- Radiosity network solver
- Combined convection-radiation solver

---

## Content Sources Mapping

| Book Section | Primary Source | Secondary Source |
|--------------|----------------|------------------|
| Part I (Intro) | Lecture 1 transcripts | New writing |
| Part II (Convection) | Lectures 2-5 transcripts | Convection tutorials, Week2-IPS |
| Part III (Conduction) | Lectures 6-8 transcripts | Conduction tutorials, Week3-IPS, Week4-IPS |
| Part IV (Radiation) | Lectures 9-11 transcripts | Radiation tutorials, Week5-IPS |

Note: Lecture 12 (phase change, special topics) can be integrated as advanced sections or appendix.

### Additional Sources (2024)

- `2024/IN-PERSON SESSIONS/` - Complex worked examples:
  - Week1-IPS: How to solve a heat transfer problem (methodology)
  - Week2-IPS: Convection worked examples
  - Week3-IPS: Conduction + Heat Exchangers & Lab
  - Week4-IPS: Extended Surfaces + **Transient (includes Age of Earth/Buffon)**
  - Week5-IPS: Radiation worked examples
  - AgeofEarth/ - Detailed Buffon experiment materials

- `2024/LECTURES/WEEK 1-6/` - Updated slide content

---

## Real-World Examples by Chapter

Curated from teaching notes - make the content engaging and memorable.

### Part I: Introduction

| Example | Concept | Source |
|---------|---------|--------|
| Sauna steam feels hotter | Heat transfer ≠ temperature | Lecture 1 transcript |
| Hot water bottles | Conduction vs convection heating | HT book note |

### Part II: Convection

| Example | Chapter | Concept |
|---------|---------|---------|
| Turkish sand coffee | Ch 4-6 | External convection, hot sand as medium |
| Lake water cooling buildings | Ch 7 | Internal flow, district cooling |
| How igloos stay warm | Ch 8 | Natural convection, enclosures |
| Coffee roasting | Ch 6 | Forced convection + conduction combined |
| Filter coffee microwave spill | Ch 7 | Superheating, nucleation |

### Part III: Conduction

| Example | Chapter | Concept |
|---------|---------|---------|
| Buffon's Age of Earth | Ch 13 | Transient cooling of sphere (full case study in Week4-IPS) |
| Thermal paste in cooking | Ch 11 | Contact resistance, fat vs lean beef |
| Glass making (Murano) | Ch 13 | Thermal shock, transient stresses |
| Hot towelettes cool fast | Ch 13 | Low thermal mass, lumped capacitance |
| Goose fat insulation layer | Ch 11 | Thermal resistance, composite layers |

### Part IV: Radiation

| Example | Chapter | Concept |
|---------|---------|---------|
| Pumpkin head shrinks overnight | Ch 15-18 | Candle radiation drying |
| LED vs incandescent spectrum | Ch 15-16 | Spectral properties, Wien's law |
| 3D-printing ovens | Ch 18 | Targeted radiative heating |

### Phase Change (Advanced/Appendix)

| Example | Concept |
|---------|---------|
| Wet bulb temperature / India heat wave | Evaporative cooling limits |
| Xiao long bao soup dumplings | Solid-liquid phase change |
| Echidnas blowing snot bubbles | Evaporative cooling in animals |
| Cooling towers | Evaporative cooling at scale |
| Tirana Skanderbeg square | Urban phase-change cooling |

---

## External Resources Policy

**Core content**: Self-contained, no embedded external videos.

**"Further Viewing" sections**: Curated links at chapter end (optional, not essential).

Rationale:
- JOSE requires "complete and immediately usable"
- YouTube links can break (link rot)
- Maintains consistent voice and quality
- External videos referenced but not embedded

Example format:
```markdown
::: {.callout-tip}
## Further Viewing
- [How Cooling Towers Work](https://youtube.com/...) - Visual explanation of evaporative cooling
- [Turkish Sand Coffee](https://youtube.com/...) - External convection in action
:::
```

---

## Chapter Template

Each chapter follows this structure:

```
# Chapter Title

## Learning Objectives
- Bullet points of what reader will learn

## Introduction
- Motivation, real-world relevance
- Where this fits in the bigger picture

## Theory
- Derivations with LaTeX equations
- Physical explanations
- Key assumptions and limitations

## Worked Examples
- Step-by-step solutions
- Python code alongside hand calculations

## Interactive Exploration
- Jupyter widgets for parameter studies
- "What happens if...?" investigations

## Summary
- Key equations (boxed)
- Key takeaways

## Exercises
- Conceptual questions
- Calculation problems
- Coding challenges

## Further Reading
- References to deeper material
```

---

## Python Code Standards

### Packages Used
- `numpy` - numerical operations
- `scipy` - ODE/PDE solvers, optimization
- `matplotlib` - static plots
- `plotly` - interactive plots (optional)
- `ipywidgets` - interactive sliders (Jupyter)

### Code Style
- Clear variable names (not single letters except standard: T, q, h, k)
- Comments explaining physics, not just code
- Functions over scripts (reusable)
- Type hints for clarity
- Docstrings with units

### Example Code Block
```python
def nusselt_flat_plate(Re: float, Pr: float, turbulent: bool = False) -> float:
    """
    Calculate Nusselt number for flow over a flat plate.

    Parameters
    ----------
    Re : float
        Reynolds number (ρVL/μ)
    Pr : float
        Prandtl number (ν/α)
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
```

---

## Visualisation Standards

All figures recreated in Python for consistency:

1. **Consistent style**: Define a style file (colors, fonts, sizes)
2. **Vector output**: Save as SVG/PDF for print quality
3. **Interactive where useful**: Temperature profiles, parameter studies
4. **Accessibility**: Colorblind-friendly palettes, sufficient contrast

### Figure Types
- **Schematics**: Simple matplotlib drawings (arrows, boxes, labels)
- **Data plots**: Line plots, contour plots, heatmaps
- **Animations**: Transient problems, flow visualisation

---

## Development Phases

### Phase 1: Skeleton
- [ ] Set up Quarto book structure
- [ ] Create chapter stubs with headings
- [ ] Define Python package/environment

### Phase 2: Content Draft
- [ ] Process transcripts into rough chapter text
- [ ] Extract key equations from slides
- [ ] Identify which figures to recreate

### Phase 3: Python Development
- [ ] Core utility functions (properties, correlations)
- [ ] Chapter-specific notebooks
- [ ] Interactive widgets

### Phase 4: Polish
- [ ] Edit text for written style
- [ ] Ensure consistent notation
- [ ] Add exercises
- [ ] Peer review

### Phase 5: Publication
- [ ] Final proofreading
- [ ] JOSE submission
- [ ] GitHub Pages deployment

---

## File Structure

```
heat-transfer-textbook/
├── _quarto.yml              # Book configuration
├── index.qmd                # Landing page
├── part-1-introduction/
│   ├── 01-what-is-heat-transfer.qmd
│   ├── 02-thermodynamic-foundations.qmd
│   └── 03-python-setup.qmd
├── part-2-convection/
│   ├── 04-boundary-layers.qmd
│   ├── 05-dimensionless-numbers.qmd
│   ├── 06-external-convection.qmd
│   ├── 07-internal-convection.qmd
│   ├── 08-natural-convection.qmd
│   └── 09-heat-exchangers.qmd
├── part-3-conduction/
│   ├── 10-fourier-law.qmd
│   ├── 11-steady-state.qmd
│   ├── 12-fins.qmd
│   ├── 13-transient.qmd
│   └── 14-multidimensional.qmd
├── part-4-radiation/
│   ├── 15-radiation-fundamentals.qmd
│   ├── 16-radiative-properties.qmd
│   ├── 17-view-factors.qmd
│   ├── 18-radiation-exchange.qmd
│   └── 19-combined-modes.qmd
├── src/
│   ├── __init__.py
│   ├── properties.py        # Fluid/material properties
│   ├── correlations.py      # Nu, f correlations
│   ├── conduction.py        # Conduction solvers
│   ├── radiation.py         # View factors, radiosity
│   └── plotting.py          # Consistent figure style
├── transcripts/             # Raw and processed transcripts
├── references/              # Bibliography
└── exercises/               # Problem sets with solutions
```

---

## Copyright and Attribution

The lectures use Incropera as a primary reference. Since this textbook is open-source (CC BY), we must ensure no copyright violation.

### Safe to Use (Not Copyrightable)

| Category | Examples | Reason |
|----------|----------|--------|
| Physical laws & equations | Fourier's law, heat equation, Navier-Stokes | Scientific facts |
| Published correlations | Dittus-Boelter, Churchill-Bernstein, etc. | Research results in public domain |
| Physical constants | σ, k values, Pr numbers | Factual data |
| Standard derivations | Fin equation, boundary layer equations | Mathematical procedures |
| General concepts | "Boundary layers form near surfaces" | Ideas aren't copyrightable |

### Must Avoid (Copyrightable)

| Category | Risk | Solution |
|----------|------|----------|
| Specific wording/paragraphs | High | Use lecture transcripts (your own words) |
| Unique figures/diagrams | High | Recreate all figures in Python |
| Exact worked examples | Medium | Design original problems with new contexts |
| Property table formatting | Low-Medium | Generate from CoolProp/NIST |

### Content Strategy

| Component | Source | Notes |
|-----------|--------|-------|
| Text explanations | Your lecture transcripts | Already in your own words |
| Correlation citations | Original research papers | More rigorous than citing textbooks |
| Property data | CoolProp (open source), NIST (public domain) | Generate tables programmatically |
| All figures | Python-generated | Solves figure copyright entirely |
| Worked examples | Original designs | Use unique contexts (cooking, climate, etc.) |

### Key Correlations - Original Sources

Cite these instead of Incropera:

- **Dittus-Boelter**: Dittus, F.W. & Boelter, L.M.K. (1930). Heat transfer in automobile radiators of the tubular type. *University of California Publications in Engineering*, 2(13), 443-461.
- **Churchill-Bernstein** (cylinder): Churchill, S.W. & Bernstein, M. (1977). A correlating equation for forced convection from gases and liquids to a circular cylinder in crossflow. *J. Heat Transfer*, 99(2), 300-306.
- **Churchill-Chu** (natural convection): Churchill, S.W. & Chu, H.H.S. (1975). Correlating equations for laminar and turbulent free convection from a vertical plate. *Int. J. Heat Mass Transfer*, 18(11), 1323-1329.

### Open-Source Property Data

```python
# Use CoolProp for fluid properties
import CoolProp.CoolProp as CP

# Example: Get water properties at 300 K
rho = CP.PropsSI('D', 'T', 300, 'P', 101325, 'Water')  # Density
k = CP.PropsSI('L', 'T', 300, 'P', 101325, 'Water')    # Conductivity
```

Add `CoolProp` to dependencies for production use.

---

## Next Steps

1. Complete video transcription
2. Set up Quarto book skeleton ✓
3. Start with Part I (Introduction) as proof of concept
4. Develop core Python utilities in `src/` ✓
5. Source original papers for all correlations used
