# Illustration Guide for Heat Transfer Textbook

This guide defines the signature visual style for all Python-generated figures in this textbook. Following these guidelines ensures visual consistency and creates a distinctive, professional appearance.

## The Signature Style

Our figures combine **editorial boldness** with **instructional clarity**, plus distinctive elements that reinforce the physics of heat transfer.

### What Makes This Style Unique

1. **Custom Thermal Colormap**: A rich 5-colour gradient (not standard matplotlib colormaps)
   - Deep blue-grey → Teal → Gold → Amber → Vibrant red
   - Represents cold-to-hot progression
   - More visually appealing than standard `coolwarm`

2. **Energy Lines**: Wavy lines near hot surfaces representing molecular vibration
   - Heat *is* molecular motion—these lines make that visible
   - Fade with distance from the heat source
   - Appear on physical system diagrams near hot boundaries

3. **Thermal Glow**: Subtle radial gradients around hot elements
   - Creates a sense of warmth emanating from hot surfaces
   - Multiple concentric layers with decreasing opacity

4. **Heat Flow Arrows**: Custom arrows with glow effect
   - Amber/orange colour (the "warm" palette colour)
   - Soft glow halo suggests energy transfer
   - Always labelled with $\dot{Q}$

5. **Thermal Scale Bar**: Thermometer-style colour reference
   - Appears on temperature plots
   - Reinforces the colour-temperature mapping
   - Provides quick visual reference

6. **Pill-Shaped Temperature Labels**: $T_1$ and $T_2$ in coloured badges
   - Red background for hot temperatures
   - Blue background for cold temperatures
   - White text, rounded corners

---

## Colour Palette

Use these exact hex codes for consistency:

```python
PALETTE = {
    # Core thermal colours (cold to hot)
    'cold': '#264653',       # Deep blue-grey
    'cool': '#2A9D8F',       # Teal
    'neutral': '#E9C46A',    # Warm gold
    'warm': '#F4A261',       # Rich amber/orange
    'hot': '#E63946',        # Vibrant red

    # UI and accent colours
    'text': '#1D3557',       # Dark blue-grey (all text)
    'accent': '#F72585',     # Magenta pink (special highlights only)
    'light_bg': '#F8F9FA',   # Plot background tint

    # Glow effects
    'glow_hot': '#FFE5E5',   # Subtle red glow
    'glow_cold': '#E5F0F8',  # Subtle blue glow
}
```

### Colour Rules

- **Hot surfaces/regions**: Always use `'hot'` (#E63946)
- **Cold surfaces/regions**: Always use `'cold'` (#264653)
- **Heat flow arrows**: Always use `'warm'` (#F4A261)
- **All text and lines**: Use `'text'` (#1D3557), never pure black
- **Plot backgrounds**: Use `'light_bg'` (#F8F9FA) for subtle warmth

---

## Custom Thermal Colormap

Create the signature colormap for temperature gradients:

```python
from matplotlib.colors import LinearSegmentedColormap

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal_signature',
    [
        (0.0, hex_to_rgb('#264653')),   # cold
        (0.25, hex_to_rgb('#2A9D8F')),  # cool
        (0.5, hex_to_rgb('#E9C46A')),   # neutral
        (0.75, hex_to_rgb('#F4A261')),  # warm
        (1.0, hex_to_rgb('#E63946')),   # hot
    ]
)
```

**Usage**: `color = THERMAL_CMAP(fraction)` where `fraction` is 0 (cold) to 1 (hot).

---

## Typography

### Font Settings

Figures use LaTeX rendering so that mathematical symbols match the MathJax-rendered body text (both use Computer Modern).

```python
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
})
```

**usetex compatibility notes:**

- Use `r'\textit{text}'` instead of `style='italic'`
- Use `r'\textbf{text}'` instead of `fontweight='bold'` on plain text
- Use `fontweight='normal'` instead of `fontweight='medium'` (usetex only supports `'normal'` and `'bold'`)
- Use `r'$^\circ$C'` for degree symbols instead of `°C`
- Newlines (`\n`) in text strings are not supported; split into separate `ax.text()` calls
- Math-mode strings (e.g. `r'$k$'`) handle their own italics; do not add `style='italic'`

### Text Styling

| Element | Size | Weight | Colour |
|---------|------|--------|--------|
| Axis labels | 11pt | medium | `'text'` |
| Tick labels | 10pt | normal | `'text'` |
| Temperature labels ($T_1$, $T_2$) | 12-15pt | bold | white on coloured pill |
| Descriptive labels ("hot surface") | 10pt | italic | matching temperature colour |
| Equations | 11pt | normal | `'text'` |
| Variable labels ($k$, $L$) | 12-15pt | bold italic | `'text'` |

---

## Figure Types and Templates

### 1. Physical System Diagrams

For slabs, walls, cylinders showing temperature distribution:

```python
# Draw gradient-filled shape
n_strips = 60
for i in range(n_strips):
    frac = i / (n_strips - 1)  # 0=cold edge, 1=hot edge
    color = THERMAL_CMAP(frac)
    # Fill strip with this colour
```

**Include**:
- Thermal gradient fill (always red=hot, teal=cold)
- Energy lines near hot surfaces (wavy lines)
- Subtle glow behind hot regions
- Pill-shaped temperature labels
- Heat flow arrow with $\dot{Q}$
- Dimension annotations
- Material property labels ($k$, $h$, etc.)

### 2. Temperature Profile Plots

For showing T vs x, T vs t, etc.:

```python
# Gradient fill under curve
for i in range(len(x) - 1):
    frac = ...  # Map position to temperature fraction
    color = THERMAL_CMAP(frac)
    ax.fill_between(x[i:i+2], baseline, y[i:i+2], color=color, alpha=0.35)

# Main line with glow
ax.plot(x, y, color=PALETTE['text'], linewidth=3,
        path_effects=[
            pe.Stroke(linewidth=5, foreground='white', alpha=0.7),
            pe.Normal()
        ])
```

**Include**:
- Gradient fill matching temperature values
- Main curve with subtle white glow
- Coloured endpoint markers (red dot at T₁, teal at T₂)
- Direct labels (no legends when possible)
- Equation box (white background, dark border)
- Thermal scale bar on right edge
- Light background tint (`'light_bg'`)

### 3. Thermal Resistance Networks

For electrical analogy circuit diagrams:

- Use standard circuit symbols (resistor zigzags)
- Colour temperature nodes: red for hot ($T_1$), teal/dark for cold ($T_2$)
- Use `'text'` colour for resistor lines and labels
- Heat flow direction indicated by arrow in `'warm'` colour

#### Fixed-Size Resistor Function

Use this function to draw resistors with a fixed zigzag body width, so all resistors look identical regardless of the distance between connection points:

```python
def _draw_resistor(ax, x0, x1, y, body_width=1.2, n_teeth=4, amp=0.2):
    """Draw a zigzag resistor symbol from (x0,y) to (x1,y).
    body_width fixes the zigzag size so all resistors look identical."""
    mid = (x0 + x1) / 2
    bx0, bx1 = mid - body_width / 2, mid + body_width / 2
    n_pts = 2 * n_teeth + 1
    xs = np.linspace(bx0, bx1, n_pts)
    ys = np.full(n_pts, float(y))
    for i in range(1, n_pts - 1):
        ys[i] = y + amp * (1 if i % 2 == 1 else -1)
    full_x = np.concatenate([[x0], xs, [x1]])
    full_y = np.concatenate([[float(y)], ys, [float(y)]])
    ax.plot(full_x, full_y, color=PALETTE['text'], linewidth=2,
            solid_capstyle='round', zorder=3)
```

The default `body_width=1.2` fits within the narrowest span (1.4 units) with 0.1-unit leads on each side.

### 4. Comparison/Multi-panel Figures

- Use consistent sizing across panels
- Label panels with (a), (b), (c) in bold
- Maintain same colour scale across related panels
- Use `layout-ncol=2` or similar in Quarto

---

## Axis and Grid Styling

```python
# Background
ax.set_facecolor(PALETTE['light_bg'])

# Grid - horizontal only, subtle
ax.yaxis.grid(True, linestyle='-', alpha=0.5, color='white', linewidth=2)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Spines - remove top and right
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(PALETTE['text'])
ax.spines['bottom'].set_color(PALETTE['text'])
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Tick styling
ax.tick_params(colors=PALETTE['text'], width=1.5)
```

---

## Signature Elements Reference

### Energy Lines Function

```python
def draw_energy_lines(ax, x_start, y_start, y_end, n_lines=5, side='left'):
    """Wavy lines representing molecular vibration near hot surfaces."""
    direction = -1 if side == 'left' else 1
    for i in range(n_lines):
        y_pos = y_start + (y_end - y_start) * (i + 0.5) / n_lines
        x_wave = np.linspace(0, 0.18, 35)
        y_wave = y_pos + 0.07 * np.sin(x_wave * 38) * (1 - x_wave/0.18)
        x_wave = x_start + direction * x_wave
        alpha = 0.45 - i * 0.07
        ax.plot(x_wave, y_wave, color=PALETTE['hot'], alpha=alpha,
                linewidth=1.8, solid_capstyle='round')
```

### Heat Flow Arrow Function

```python
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

def draw_heat_arrow(ax, start, end, label=None):
    """Heat flow arrow with glow effect."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=3,
        color=PALETTE['warm'],
        path_effects=[
            pe.Stroke(linewidth=6, foreground=PALETTE['glow_hot'], alpha=0.4),
            pe.Normal()
        ],
        zorder=10
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2 + 0.12
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=14,
                fontweight='bold', color=PALETTE['warm'])
```

### Thermal Scale Bar Function

```python
def draw_thermal_scale(ax, x, y_bottom, height, t_min, t_max):
    """Thermometer-style colour scale."""
    width = 0.05
    n_segments = 50
    for i in range(n_segments):
        y0 = y_bottom + i * height / n_segments
        y1 = y_bottom + (i + 1) * height / n_segments
        frac = i / n_segments
        color = THERMAL_CMAP(frac)
        ax.fill([x, x, x + width, x + width], [y0, y1, y1, y0],
                color=color, edgecolor='none')
    # Add border and labels...
```

---

## Figure Sizes

| Type | figsize | Use case |
|------|---------|----------|
| Single diagram | `(6, 5)` | Physical system only |
| Diagram + plot | `(12, 5)` | Side-by-side layout |
| Full-width plot | `(10, 4)` | Data visualization |
| Three panels | `(14, 4)` | Comparison figures |
| Small inline | `(4, 3)` | Supporting figures |

---

## Export Settings

```python
plt.savefig('figure_name.png',
            dpi=200,                    # High quality
            facecolor='white',          # White background
            bbox_inches='tight',        # Crop whitespace
            pad_inches=0.1)             # Small padding
```

For PDF output (vector graphics):
```python
plt.savefig('figure_name.pdf',
            facecolor='white',
            bbox_inches='tight')
```

---

## MANDATORY: Overlap Detection and Figure Quality

**Every figure must include an overlap check AND a visual inspection of the rendered output.** The overlap checker catches most issues, but it is not a substitute for looking at the image.

### The Golden Rule

**Always start from `signature_style_final.py` as the reference implementation.** Do not freelance layouts. Copy the positioning, spacing, and structure from the reference, then adapt for the specific figure content. This prevents the majority of overlap and layout issues.

### Overlap Checker (corrected version)

**Critical:** `text.get_window_extent()` only measures the text itself — it does NOT include `bbox` padding (e.g. pill label backgrounds with `pad=0.3`). You MUST use `get_bbox_patch().get_window_extent()` for text with bbox styling, otherwise pill labels will appear to not overlap when they visually do.

Add this function and call it at the end of every figure code block:

```python
import warnings

def _check_overlaps(fig):
    """Automatic overlap detection for all text elements.
    Uses bbox patch extent for pill-styled labels (includes padding)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for ax in fig.axes:
        texts = [(t, t.get_text()) for t in ax.texts
                 if t.get_text().strip() and t.get_visible()]
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if lbl.get_text().strip() and lbl.get_visible():
                texts.append((lbl, lbl.get_text()))
        bboxes = []
        for t, name in texts:
            try:
                # Use bbox patch if present (includes pill padding)
                bp = t.get_bbox_patch()
                if bp is not None:
                    bboxes.append((bp.get_window_extent(renderer), name[:30]))
                else:
                    bboxes.append((t.get_window_extent(renderer), name[:30]))
            except Exception:
                pass
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bboxes[i][0].overlaps(bboxes[j][0]):
                    issues.append(f"'{bboxes[i][1]}' ↔ '{bboxes[j][1]}'")
    if issues:
        warnings.warn("OVERLAPPING LABELS:\n  " + "\n  ".join(issues))
    return issues

# Call at the end of every figure, before plt.show()
overlaps = _check_overlaps(fig)
if overlaps:
    print(f"WARNING: {len(overlaps)} overlapping label(s)")
```

### Using `figure_utils` (for standalone scripts)

```python
from style_experiments.figure_utils import check_overlaps, check_and_save

# Check only
fig.canvas.draw()
overlaps = check_overlaps(fig)

# Check and save in one step (raises error if overlaps found)
check_and_save(fig, 'output.png', raise_on_overlap=True)
```

### Pill Label Spacing Rules

Pill labels (`bbox=dict(boxstyle='round,pad=0.3', ...)`) are visually much larger than plain text. The `pad=0.3` adds significant space around the text that is invisible to naive extent checks. Follow these exact rules:

1. **Stack vertically, never side-by-side**: Place pills and their description text in vertical stacks (e.g. pill at y=4.45, description at y=4.85)
2. **Minimum 0.55 data units between pill center and adjacent text**: The pill's padding extends ~0.15–0.2 units beyond center, so 0.4 units between centers is NOT enough — use 0.55
3. **Asymmetric compensation**: A pill ABOVE a description needs ~0.4 units gap (pill bottom to text top). A pill BELOW a description needs ~0.55 units gap because the pill background extends further visually in the downward direction
4. **Reference positions from `signature_style_final.py`**:
   - T₁ pill at y=4.45, "hot surface" at y=4.85 (0.4 gap, pill is below text)
   - T₂ pill at y=−0.4, "cold surface" at y=−0.95 (0.55 gap, pill is above text)

### Figure Construction Rules

1. **Use `fig.add_axes([left, bottom, width, height])`** for precise panel positioning — not `plt.subplots()` which gives less control
2. **Vertical slab orientation** (hot=top, cold=bottom) is the standard layout for physical system diagrams
3. **Include ALL signature elements**: shadow, thermal glow (Circle patches), energy lines, pill labels, thermal scale bar, FancyBboxPatch for equations, annotation leader lines
4. **Extend axis limits** generously — leave buffer around all labels (`ylim` should accommodate pill + description + margin)

### Post-Generation Verification

The overlap checker is necessary but NOT sufficient. After rendering every figure:

1. Run `_check_overlaps(fig)` — fix any reported overlaps
2. Render with `quarto render`
3. **Visually inspect the output PNG** — check for:
   - Labels that are technically not overlapping but visually too close
   - Text obscured by energy lines or gradient fills
   - Elements clipped by axis limits
   - Asymmetric spacing that looks unbalanced

### Common Fixes

| Problem | Fix |
|---------|-----|
| Pill overlaps adjacent text | Increase gap to ≥0.55 data units |
| `get_window_extent` misses pill size | Use `get_bbox_patch().get_window_extent()` |
| Top and bottom spacing look uneven | Add 0.15 extra gap below pills (padding asymmetry) |
| Text obscured by energy lines | Move text outside the slab y-range |
| Label clipped by axis | Extend `xlim`/`ylim` |

---

## Checklist for Each Figure

- [ ] **Based on `signature_style_final.py`** — layout copied from reference, not freelanced
- [ ] Uses `fig.add_axes()` for panel positioning
- [ ] Uses signature thermal colormap (not matplotlib defaults)
- [ ] Colour direction correct: red=hot, teal=cold
- [ ] Text colour is `'text'` (#1D3557), not black
- [ ] Temperature labels have coloured pill backgrounds
- [ ] Pill-to-text spacing ≥ 0.55 data units (accounting for bbox padding)
- [ ] Hot surfaces have energy lines (if physical diagram)
- [ ] Heat flow arrows use `'warm'` colour with glow
- [ ] Shadow behind physical diagram elements
- [ ] Thermal glow (Circle patches) near hot surfaces
- [ ] Plot has light background tint
- [ ] Horizontal grid only, subtle white lines
- [ ] Top and right spines removed
- [ ] Equation in FancyBboxPatch (not just bbox on text)
- [ ] Thermal scale bar on temperature plots
- [ ] Annotation leader lines for T₁/T₂ labels on plots
- [ ] **Overlap check passes** — `_check_overlaps(fig)` with bbox-aware checker
- [ ] **Visually inspected** — rendered PNG reviewed for spacing, readability, balance
- [ ] `code-fold: true` in Quarto to hide code by default

---

## Example: Complete Figure Template

**Always start from `style_experiments/signature_style_final.py`** and adapt. See `part-1-introduction/02-thermodynamic-foundations.qmd` (fig-1d-slab) for a complete working example embedded in a Quarto chapter.

```python
#| label: fig-example
#| fig-cap: "Caption describing the figure and its key insight."
#| code-fold: true

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
})

# [Include PALETTE and THERMAL_CMAP definitions]
# [Include _check_overlaps function (bbox-aware version)]
# [Use fig.add_axes() for layout — copy from signature_style_final.py]
# [Include ALL signature elements: shadow, glow, energy lines, pills, scale bar]
# [Call _check_overlaps(fig) before plt.show()]
```
