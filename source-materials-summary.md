# Source Materials Summary

Analysis of the Thermofluids teaching materials (excluding THERMOFLUIDS3-HEAT-TRANSFER)

## Overview

~3,800 files spanning 2020-2024, with comprehensive lecture videos, slides, tutorials, and lab materials.

## Directory Structure

```
Thermofluids/
├── 2020/                           (Fluid mechanics + turbomachinery focus)
├── 2021/                           (Most comprehensive - heat transfer course)
│   ├── CAPTURES/                   (Video lectures by week)
│   ├── LECTURES/                   (Keynote slides by week)
│   ├── TUTORIALS/                  (Problem sets with LaTeX)
│   ├── LAB/
│   ├── EXAM/
│   └── FORMULA SHEET/
├── 2022/
├── 2023/
├── 2024/                           (Most recent iteration)
└── THERMOFLUIDS/
    └── transferHeatTransferForRohit/   (44 older recorded lectures)
```

## File Types

| Type | Count | Purpose |
|------|-------|---------|
| PDF | ~1,400 | Lecture slides, problems, solutions |
| Keynote (.key) | ~350 | Presentation source files |
| LaTeX (.tex) | ~285 | Tutorial exercises, formula sheets |
| Video (.mp4) | ~238 | Recorded lectures |
| **Python/Jupyter** | **0** | **Opportunity for textbook** |

## Lecture Slide Formats

Each lecture exists in multiple formats:

```
WEEK X/
├── LECTURE Y/
│   ├── Y.1.Topic.key          (Keynote - individual section)
│   ├── Y.2.Topic.key
│   └── Y.FULL.key             (Keynote - complete lecture)
├── PDFs/
│   ├── Y.1.Topic.pdf          (PDF - individual section)
│   ├── Y.2.Topic.pdf
│   └── ...
├── LECTURE_Y_FULL.pdf         (PDF - complete lecture)
└── Y.Quiz.docx
```

**For the textbook**: The PDF versions are more portable and can be used directly as reference material.

## Primary Video Source: 2021/CAPTURES

Organised by week with ~70+ lecture segments:

| Week | Videos | Topics Covered |
|------|--------|----------------|
| WEEK1 | 17 | Welcome, Parts 1.1-1.5, 2.1-2.6 (Fundamentals, Boundary Layers) |
| WEEK2 | 11 | Parts 3.1-3.6, 4.1-4.6 (Similarity, Nu Correlations, Internal Flows) |
| WEEK3 | 11 | Parts 5.1-5.5, 6.1-6.6 (Heat Exchangers, Conduction) |
| WEEK4 | 11 | Parts 7.1-7.5, 8.1-8.6 (Fins, Transient Conduction) |
| WEEK5 | 14 | Parts 9.0-9.6, 10.0-10.5 (Radiation Properties, View Factors) |
| WEEK6 | 7 | Parts 11.1-11.3, 12.1-12.4 (Combined Modes, Phase Change) |

## Lecture Topics (from 2021/LECTURES)

### Lecture 1 - Fundamentals
- 1.1 Basics
- 1.2 Intro-Conduction
- 1.3 Intro-Resistances
- 1.4 Intro-Convection
- 1.5 Intro-Radiation

### Lecture 2 - Boundary Layers
- 2.1-2.2 Boundary layer basics
- 2.3 Deriving continuity equation
- 2.4 Deriving momentum equations
- 2.5 Deriving energy equation
- 2.6 Simplifying equations

### Lecture 3 - External Convection
- 3.1 Similarity solutions
- 3.2 What are Nu correlations
- 3.3 Worked example: flat plate
- 3.4 Other engineering flows
- 3.5 Free convection

### Lecture 4 - Internal Convection
- 4.1 What are internal flows
- 4.2-4.4 Heat transfer in pipe flows
- 4.5 Worked example: flow steriliser
- 4.6 Worked example: CSP generation

### Lecture 5 - Heat Exchangers
- 5.1 Introduction and thermal modelling
- 5.2 Types of heat exchangers
- 5.3 Effectiveness and NTU
- 5.4-5.5 Worked examples

### Lecture 6 - Conduction Fundamentals
- 6.1 Intro and overview
- 6.2 Deriving diffusion equation
- 6.3 Using derived equation
- 6.4 Thermal circuits
- 6.5 Contact and interfacial resistances
- 6.6 Worked examples

### Lecture 7 - Extended Surfaces (Fins)
- 7.1 Deriving a fin equation
- 7.2 Solving fin equations
- 7.3 Worked examples
- 7.4 Realistic fins and fin arrays
- 7.5 Worked example: cylindrical sleeve

### Lecture 8 - Advanced Conduction
- 8.1 Intro and analytical solutions
- 8.2 Conduction shape factors
- 8.3 Introducing transient conduction
- 8.4 Lumped capacitance
- 8.5 Analytical solutions
- 8.6 Numerical methods

### Lecture 9 - Radiation Properties
- 9.0 How does a radiometer work
- 9.1 Re-introduction to radiation
- 9.2 Spectral properties of radiation
- 9.3 Spatial properties of radiation
- 9.4 Radiative heat fluxes
- 9.5 Deriving Kirchhoff's law

### Lecture 10 - Radiation Exchange
- 10.0 Radiation and the climate
- 10.1 View factors
- 10.2 Exchange between blackbodies
- 10.3 Exchange between non-blackbodies
- 10.4-10.5 Worked examples (including thermos flask)

### Lecture 11 - Combined Modes
- 11.1 Introduction and overview
- 11.2 Spectral radiation with convection
- 11.3 Radiative circuits with convection

### Lecture 12 - Special Topics
- 12.1 Convection with phase change
- 12.2 Critical insulation thickness
- 12.3-12.4 Internal heat in a journal bearing

## Tutorial Problem Sets

Located in `2021/TUTORIALS/` with LaTeX sources:

| Topic | Folder | Contents |
|-------|--------|----------|
| Introductory | `Introductory-LaTeX/` | Basic heat transfer concepts |
| Conduction | `Conduction-LaTeX/` | 1D/multi-D conduction, fins |
| Convection | `Convection-LaTeX/` | External/internal flow correlations |
| Radiation | `Radiation-LaTeX/` | Radiation properties, view factors |

Each folder contains:
- `tutorials.tex` - LaTeX source
- `tutorials.pdf` - Compiled problem set
- Supporting images (PNG files)

## Lab Materials

Located in `2024/LAB/`:
- `lab.tex` / `lab.pdf` - Lab handout (heat exchanger experiment)
- `Lab-procedure.pdf` - Step-by-step guide
- `Lab2_Heat_Exchanger*.xlsx` - Real experimental data
- `SAMPLE_SOLNS/` - Graded examples at different levels

## Additional Resources

### 2020/LECTURES (Related Fluid Mechanics Content)
- Viscous flows (1-4)
- Compressible flows (1-2)
- Turbomachinery (1-5)
- Fluid mechanics keynotes

### THERMOFLUIDS/transferHeatTransferForRohit
- 44 older video lectures (2019)
- Lecture slides
- Tutorials with solutions
- Historical reference material

### Formula Sheets
- `2021/FORMULA SHEET/` and `2024/FORMULA SHEET/`
- Comprehensive reference with correlations, properties, charts

## Key Resource Paths

| Resource | Best Source |
|----------|-------------|
| Video lectures | `2021/CAPTURES/WEEK1-6/` |
| Keynote slides | `2021/LECTURES/WEEK 1-6/` |
| Tutorials (LaTeX) | `2021/TUTORIALS/*-LaTeX/` |
| Lab materials | `2024/LAB/` |
| Formula sheet | `2024/FORMULA SHEET/` |
| Exams | `2024/EXAM/` |

## Suggested Python Code Examples

Based on lecture content, high-value computational additions:

### Convection (Lectures 2-4)
- Boundary layer thickness calculator
- Nusselt correlation lookup (flat plate, cylinder, sphere, pipe)
- Reynolds/Prandtl number explorer
- Pipe flow heat transfer solver

### Heat Exchangers (Lecture 5)
- LMTD method calculator
- Effectiveness-NTU solver
- Heat exchanger sizing/rating tool

### Conduction (Lectures 6-8)
- 1D steady conduction solver
- Thermal resistance network builder
- Fin efficiency calculator
- Transient conduction (lumped capacitance checker)
- Numerical solution of heat equation

### Radiation (Lectures 9-11)
- Blackbody spectrum plotter (Planck's law)
- View factor calculator (common geometries)
- Radiosity method for enclosures
- Combined convection-radiation solver

### Special Topics (Lecture 12)
- Critical insulation thickness calculator
- Phase change heat transfer examples

## Next Steps

1. Set up Quarto book structure
2. Start with Lecture 1 content (Fundamentals + environment setup)
3. Add Lecture 2-4 (Convection) as first major chapter
4. Convert worked examples to Jupyter notebooks
5. Create interactive Python versions of key calculations
