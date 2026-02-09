# Writing Guidelines for This Textbook

These guidelines ensure consistency across all chapters.

## 0. The Core Balance

The textbook should achieve a specific balance:

- **Personal voice**: Sounds like the author teaching, with real examples from lectures, first-person perspective where appropriate, and the author's characteristic explanations
- **Technical rigour**: Proper definitions with units, symbol tables, dimensional analysis, quantitative examples, and citations for claims

This is NOT a dry reference text, but it's also NOT a casual blog post. It should read like an excellent lecturer wrote a proper textbook in their own voice. When in doubt, ask: "Does this sound like me, AND is it technically complete?"

Specific markers of this balance:

- Opening with a concrete example or demonstration (Stirling engine, sauna, etc.)
- First-person where it adds authenticity ("I have a Stirling engine on my desk")
- But then rigorous treatment: equations with symbol tables, unit checking, proper definitions
- Anecdotes and examples are supported with evidence (links to sources, quantitative data)
- Exercises that range from conceptual to computational

## 1. Use Lecture Transcripts as Primary Source

Always use the lecture transcripts (`transcripts/raw/`) as the foundation for chapter content. The workflow is:

1. Read the relevant transcript(s) for a chapter
2. Extract explanations, anecdotes, and phrasing in the author's voice
3. Edit spoken → written style (remove filler words, fix run-on sentences)
4. Reorganize for textbook flow
5. Add equations, figures, Python code around the core text

This ensures the textbook is *my* textbook, not a generic one. The transcripts contain my teaching style, my examples, my way of explaining things.

## 2. Push Back When Appropriate

The goal is the best possible textbook, not validation of the author's biases. If something in the transcripts or author instructions could be improved, say so. Don't just ignore concerns—raise them—but also don't ignore the source material.

## 3. Write Like a Human

Avoid typical LLM writing signatures:

- Excessive em-dashes
- "Let me explain...", "It's worth noting...", "Importantly,..."
- Rhetorical questions followed by immediate answers
- Repetitive sentence structures
- Over-hedging and excessive qualifiers

Check sources when unsure about facts. Use natural, varied prose.

## 4. Adapt for the Book Medium

Transcripts are spoken; books are read. Adapt accordingly:

### Formatting

- **Bold** for key terms when first introduced (e.g., **thermal conductivity**, **boundary layer**)
- *Italics* for emphasis and for "when you hear X, think Y" constructions
- Bullet points for lists of 3+ items
- Numbered lists for sequential steps or procedures
- Tables for comparing quantities, listing symbols with units, or showing relationships

### Equations

After each governing equation, include a **symbol table** with:

| Symbol | Name | SI Units | Notes (if needed) |
|--------|------|----------|-------------------|

Example:
```markdown
| Symbol | Name | SI Units |
|--------|------|----------|
| $q''$ | Heat flux | W/m² |
| $k$ | Thermal conductivity | W/(m·K) |
| $dT/dx$ | Temperature gradient | K/m |
```

### Quotes and Callouts

Use callout boxes for:

- **Famous quotes** (use `{.callout-note appearance="simple"}` with blockquote inside)
- **Key insights** that readers should remember
- **Warnings** about common mistakes
- **Course structure** notes

Example for quotes:
```markdown
::: {.callout-note appearance="simple"}
> "Everything that living things do can be understood in terms of the jiggling of atoms."
>
> — Richard Feynman
:::
```

### Depth and Rigour

- Include dimensional analysis with actual math, not just assertions
- Show unit checking explicitly: $[\text{W/m}^2] = [\text{W/(m·K)}] \times [\text{K/m}]$
- Provide quantitative examples (costs, temperatures, percentages)
- Link claims to evidence (cite sources for anecdotes)

### Avoid Redundancy Between Chapters

- Do not repeat definitions or explanations that appeared in earlier chapters
- Instead, reference the earlier chapter: "Recall from Chapter 1 that..."
- Only repeat content when pedagogically necessary (e.g., a brief reminder before building on it)
- Exception: Key equations may be restated if needed for a derivation, but don't re-explain notation

For example, if Chapter 1 defines the prime notation ($\dot{Q}$, $q'$, $q''$), Chapter 2 should reference it rather than repeat the full explanation.

## 5. Images and Figures

### Every Chapter Needs Images

Chapters without images feel incomplete. For each chapter, include:

- **At least 2-3 figures** illustrating key concepts
- **Diagrams** for physical setups (control volumes, geometries, circuits)
- **Plots** from Python code where appropriate
- **Photos** from lecture slides for real-world examples

Sources for images:
1. Extract from lecture slides (PDF) using PyMuPDF
2. Generate with Python/Matplotlib
3. Create simple diagrams describing the physics

When writing a chapter, identify image opportunities:
- Physical systems being analysed (e.g., "heat through a wall" → draw the wall with temperatures labelled)
- Analogies (e.g., thermal-electrical → draw both circuits side by side)
- Real-world applications (e.g., insulation materials, heat exchangers)
- Graphs showing relationships (e.g., temperature profiles, resistance vs thickness)

### Python Figure Style — The Signature Style

All Python-generated figures must follow the **signature style** defined in [`ILLUSTRATION_GUIDE.md`](ILLUSTRATION_GUIDE.md). This creates a distinctive, consistent visual identity across the textbook.

**Key signature elements:**

1. **Custom thermal colormap**: Deep blue → teal → gold → amber → red (not standard matplotlib colormaps)
2. **Energy lines**: Wavy lines near hot surfaces representing molecular vibration
3. **Thermal glow**: Subtle radial gradients around hot elements
4. **Heat flow arrows**: Amber arrows with glow effect, labelled with $\dot{Q}$
5. **Pill-shaped labels**: $T_1$ in red badge, $T_2$ in blue badge
6. **Thermal scale bar**: Thermometer-style colour reference on temperature plots

**Quick reference:**

| Colour | Hex | Use |
|--------|-----|-----|
| Hot | `#E63946` | Hot surfaces, T₁ labels |
| Warm | `#F4A261` | Heat flow arrows |
| Neutral | `#E9C46A` | Mid-temperature |
| Cool | `#2A9D8F` | Transition colour |
| Cold | `#264653` | Cold surfaces, T₂ labels |
| Text | `#1D3557` | All text and lines |

See `ILLUSTRATION_GUIDE.md` for complete implementation details, helper functions, and templates.

Always use `code-fold: true` to hide code by default, showing only the figure

### Sizing

- When multiple images appear in a layout, use consistent heights
- Use `height=` rather than `width=` when images should match vertically
- For side-by-side figures: `layout-ncol=2` with matching height specifications

Example:
```markdown
::: {#fig-example layout-ncol=2}

![Caption A](image-a.png){height=250px}

![Caption B](image-b.png){height=250px}

Overall caption for the figure group.
:::
```

### Captions

- Captions should be informative, not just descriptive
- Include the key takeaway if possible

## 6. References and Further Viewing

### Inline Links

When referencing external resources (videos, articles), include them **inline** at the relevant point in the text, not just at the end. Use a linked note or parenthetical:

```markdown
The no-slip condition means fluid velocity is zero at a solid surface. (See [this visualisation of boundary layers](https://youtube.com/...).)
```

Or as a margin note / aside if the format supports it.

### Evidence for Anecdotes

Historical anecdotes and claims should link to sources:

```markdown
The Mars Climate Orbiter was lost due to a unit mismatch, costing $327.6 million. [@nasa-mco-report]
```

Or inline: `[NASA investigation report](https://url)`

### Further Viewing Section

Keep the collapsible "Further Viewing" section at chapter end, but ensure each resource is also referenced inline where relevant.

## 7. Exercises

Structure exercises with:

- **Bold descriptive titles** for each problem
- A mix of:
  - Conceptual questions (identify modes, explain phenomena)
  - Quantitative problems (calculate values, check units)
  - Python/computational tasks (modify code, plot relationships)
- Unit analysis problems to reinforce dimensional consistency

## 8. Interactive Python Code

Use `{pyodide-python}` code blocks for interactive examples that readers can edit and run in the browser. The quarto-pyodide extension enables this.

### Browser-Based Execution

The book uses [quarto-pyodide](https://github.com/coatless-quarto/pyodide) to run Python directly in the browser via WebAssembly. Readers don't need to install anything to experiment with code.

Mention this in the preface and remind readers when they encounter their first interactive block.

### When to Use Interactive Code

- **Exploratory examples**: "Try changing X to see what happens"
- **Parameter studies**: Varying inputs to observe effects
- **Building intuition**: "What happens when radiation equals convection?"

### Formatting Interactive Blocks

```markdown
```{pyodide-python}
# ===== TRY CHANGING THESE VALUES =====
T_s = 400      # Surface temperature (K)
T_inf = 300    # Ambient temperature (K)
# =====================================

# ... rest of code
```
```

Key points:

- Highlight editable parameters with comment blocks
- Keep code concise (Pyodide runs in browser, so avoid heavy computation)
- Add a print statement showing which mode/value dominates
- Note in the text that the code is interactive: "**This code is interactive**: try changing..."

### Static vs Interactive

Use regular `{python}` blocks for:

- Complex visualizations with matplotlib (Pyodide support is limited)
- Code that takes a long time to run
- Examples where interactivity doesn't add value

Use `{pyodide-python}` for:

- Parameter exploration
- Simple calculations readers should experiment with
- Building intuition through "what if" scenarios

## 9. Copyright Strategy

- **Text**: From transcripts (author's own words)
- **Equations**: Scientific facts (not copyrightable)
- **Correlations**: Cite original research papers, not Incropera
- **Figures**: All Python-generated or author-created
- **Properties**: CoolProp/NIST (open source)
- **Examples**: Original contexts from teaching

## 10. Chapter Structure Template

Each chapter should follow this structure:

```
# Chapter Title

::: {.callout-note}
## Learning Objectives
- Bullet points
:::

## Introduction / Motivation
[Connect to prior knowledge, real-world relevance]

## Main Content Sections
[Theory with equations, symbol tables, worked examples]
[Inline references to videos/resources where relevant]

## Summary

::: {.callout-tip}
## Key Takeaways
- Bullet points with **bold** key terms
:::

::: {.callout-important}
## Key Equations
[Boxed equations]
:::

## Exercises
[Numbered, with bold titles]

::: {.callout-tip collapse="true"}
## Further Viewing
[Curated links - but also referenced inline above]
:::
```
