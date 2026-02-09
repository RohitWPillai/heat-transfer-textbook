"""
FINAL SIGNATURE STYLE for Heat Transfer Textbook
Fixed: colour direction (red=hot at top, blue=cold at bottom)
Fixed: label overlap on temperature axis

This style is designed to be universal across all figure types in the book.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# Import overlap checker
from figure_utils import check_and_save, check_overlaps

# =============================================================================
# SIGNATURE COLOUR PALETTE
# =============================================================================
PALETTE = {
    # Core thermal colours
    'hot': '#E63946',        # Vibrant warm red
    'warm': '#F4A261',       # Rich amber/orange
    'neutral': '#E9C46A',    # Warm gold (transition)
    'cool': '#2A9D8F',       # Teal (transition)
    'cold': '#264653',       # Deep blue-grey

    # Accent and UI colours
    'accent': '#F72585',     # Magenta pink (special highlights)
    'text': '#1D3557',       # Dark blue-grey for text
    'light_bg': '#F8F9FA',   # Subtle warm white background

    # Glow effects
    'glow_hot': '#FFE5E5',   # Subtle red glow
    'glow_cold': '#E5F0F8',  # Subtle blue glow
}

def hex_to_rgb(hex_color):
    """Convert hex to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

# Create custom thermal colormap: cold (0) -> hot (1)
thermal_colors = [
    (0.0, hex_to_rgb(PALETTE['cold'])),
    (0.25, hex_to_rgb(PALETTE['cool'])),
    (0.5, hex_to_rgb(PALETTE['neutral'])),
    (0.75, hex_to_rgb(PALETTE['warm'])),
    (1.0, hex_to_rgb(PALETTE['hot']))
]
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal_signature',
    [(pos, color) for pos, color in thermal_colors]
)


# =============================================================================
# SIGNATURE VISUAL ELEMENTS
# =============================================================================

def draw_energy_lines(ax, x_start, y_start, y_end, n_lines=5, side='left'):
    """
    Draw wavy 'energy/vibration' lines near hot surfaces.
    These represent molecular motion - the essence of heat.
    """
    direction = -1 if side == 'left' else 1

    for i in range(n_lines):
        y_pos = y_start + (y_end - y_start) * (i + 0.5) / n_lines

        # Create wavy line that fades out
        x_wave = np.linspace(0, 0.18, 35)
        y_wave = y_pos + 0.07 * np.sin(x_wave * 38) * (1 - x_wave/0.18)
        x_wave = x_start + direction * x_wave

        # Fade alpha with distance from surface
        alpha = 0.45 - i * 0.07
        ax.plot(x_wave, y_wave, color=PALETTE['hot'], alpha=alpha,
                linewidth=1.8, solid_capstyle='round')


def draw_heat_arrow(ax, start, end, label=None):
    """Draw a distinctive gradient heat flow arrow with glow effect."""
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
                fontweight='bold', color=PALETTE['warm'],
                ha='left', va='center')


def draw_thermal_scale(ax, x, y_bottom, height, t_min, t_max):
    """Draw a thermometer-style thermal scale bar."""
    width = 0.05
    n_segments = 50

    # Draw gradient bar (cold at bottom, hot at top)
    for i in range(n_segments):
        y0 = y_bottom + i * height / n_segments
        y1 = y_bottom + (i + 1) * height / n_segments
        frac = i / n_segments  # 0 at bottom (cold), 1 at top (hot)
        color = THERMAL_CMAP(frac)
        ax.fill([x, x, x + width, x + width], [y0, y1, y1, y0],
                color=color, edgecolor='none')

    # Border
    ax.plot([x, x], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.2)
    ax.plot([x + width, x + width], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.2)

    # Tick marks and labels
    for frac, temp in [(0, t_min), (0.5, (t_min+t_max)/2), (1, t_max)]:
        y = y_bottom + frac * height
        ax.plot([x + width, x + width + 0.015], [y, y],
                color=PALETTE['text'], linewidth=1)
        ax.text(x + width + 0.025, y, f'{temp:.0f}°C', fontsize=8,
                va='center', color=PALETTE['text'])


# =============================================================================
# MAIN FIGURE - CORRECTED
# =============================================================================

def create_signature_figure():
    """Create the final signature style figure."""

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Layout
    ax1 = fig.add_axes([0.05, 0.12, 0.38, 0.80])   # Physical diagram
    ax2 = fig.add_axes([0.50, 0.12, 0.44, 0.80])   # Temperature plot

    T1, T2 = 100, 20  # Hot and cold temperatures

    # =========================================================================
    # LEFT: Physical System Diagram
    # =========================================================================
    ax1.set_xlim(-0.4, 1.7)
    ax1.set_ylim(-1.1, 5.2)
    ax1.set_facecolor('white')

    # Subtle glow behind hot region
    for r, alpha in [(0.6, 0.08), (0.4, 0.12), (0.25, 0.15)]:
        circle = Circle((0.65, 4.2), r, facecolor=PALETTE['glow_hot'],
                        edgecolor='none', alpha=alpha, zorder=0)
        ax1.add_patch(circle)

    # Energy/vibration lines near hot surface
    draw_energy_lines(ax1, 0.25, 3.6, 4.3, n_lines=4, side='left')

    # Draw slab with thermal gradient
    # FIXED: hot (red) at TOP, cold (blue) at BOTTOM
    n_strips = 60
    slab_bottom, slab_top = 0.2, 4.0
    slab_left, slab_right = 0.25, 1.05

    for i in range(n_strips):
        y0 = slab_bottom + i * (slab_top - slab_bottom) / n_strips
        y1 = slab_bottom + (i + 1) * (slab_top - slab_bottom) / n_strips
        # frac goes from 0 (bottom=cold) to 1 (top=hot)
        frac = i / (n_strips - 1)
        color = THERMAL_CMAP(frac)
        ax1.fill([slab_left, slab_left, slab_right, slab_right],
                 [y0, y1, y1, y0], color=color, edgecolor='none')

    # Slab border with subtle shadow
    shadow_offset = 0.025
    ax1.plot([slab_left + shadow_offset, slab_left + shadow_offset,
              slab_right + shadow_offset, slab_right + shadow_offset],
             [slab_bottom - shadow_offset, slab_top - shadow_offset,
              slab_top - shadow_offset, slab_bottom - shadow_offset],
             color='#00000015', linewidth=8, solid_capstyle='round', zorder=1)

    ax1.plot([slab_left, slab_left, slab_right, slab_right, slab_left],
             [slab_bottom, slab_top, slab_top, slab_bottom, slab_bottom],
             color=PALETTE['text'], linewidth=2.5, zorder=2)

    # Temperature labels
    # T1 (hot) at top - red pill
    ax1.text(0.65, 4.45, r'$T_1$', fontsize=15, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['hot'],
                      edgecolor='none'), zorder=10)
    ax1.text(0.65, 4.8, 'hot surface', fontsize=10, ha='center',
             color=PALETTE['hot'], style='italic')

    # T2 (cold) at bottom - blue pill with proper spacing
    ax1.text(0.65, -0.4, r'$T_2$', fontsize=15, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['cold'],
                      edgecolor='none'), zorder=10)
    ax1.text(0.65, -0.8, 'cold surface', fontsize=10, ha='center',
             color=PALETTE['cold'], style='italic')

    # Heat flow arrow
    draw_heat_arrow(ax1, (1.35, 3.5), (1.35, 0.7), r'$\dot{Q}$')

    # Thickness dimension (moved to right side of slab)
    ax1.annotate('', xy=(1.15, slab_top), xytext=(1.15, slab_bottom),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.3,
                               color=PALETTE['text'], mutation_scale=8))
    ax1.text(1.22, (slab_top + slab_bottom)/2, r'$L$', fontsize=12, ha='left',
             va='center', color=PALETTE['text'], fontweight='medium')

    # Material property label
    ax1.text(0.10, 2.1, 'Thermal\nconductivity', fontsize=9, ha='right',
             va='center', color=PALETTE['text'], linespacing=1.3)
    ax1.text(0.10, 1.65, r'$k$', fontsize=15, ha='right', va='center',
             color=PALETTE['text'], fontweight='bold', style='italic')

    ax1.axis('off')

    # =========================================================================
    # RIGHT: Temperature Profile Plot
    # =========================================================================
    ax2.set_facecolor(PALETTE['light_bg'])

    x = np.linspace(0, 1, 100)
    T = T1 - (T1 - T2) * x  # Linear: T1 at x=0, T2 at x=1

    # Gradient fill under curve
    # FIXED: matches slab - red at x=0 (T1=hot), blue at x=1 (T2=cold)
    for i in range(len(x) - 1):
        frac = 1 - x[i]  # 1 at x=0 (hot), 0 at x=1 (cold)
        color = THERMAL_CMAP(frac)
        ax2.fill_between(x[i:i+2], 0, T[i:i+2], color=color, alpha=0.35)

    # Main line with subtle glow
    ax2.plot(x, T, color=PALETTE['text'], linewidth=3,
             path_effects=[
                 pe.Stroke(linewidth=5, foreground='white', alpha=0.7),
                 pe.Normal()
             ], zorder=5)

    # Endpoint markers
    ax2.scatter([0], [T1], s=110, color=PALETTE['hot'], zorder=10,
                edgecolors='white', linewidths=2.5)
    ax2.scatter([1], [T2], s=110, color=PALETTE['cold'], zorder=10,
                edgecolors='white', linewidths=2.5)

    # Labels positioned to avoid overlap - use annotation lines
    ax2.annotate(r'$T_1$', xy=(0, T1), xytext=(0.08, T1 + 8),
                fontsize=12, ha='left', va='bottom',
                color=PALETTE['hot'], fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=PALETTE['hot'], lw=1))
    ax2.annotate(r'$T_2$', xy=(1, T2), xytext=(0.92, T2 - 8),
                fontsize=12, ha='right', va='top',
                color=PALETTE['cold'], fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=PALETTE['cold'], lw=1))

    # Equation box
    eq_box = FancyBboxPatch(
        (0.26, 42), 0.48, 22,
        boxstyle='round,pad=0.02,rounding_size=0.03',
        facecolor='white',
        edgecolor=PALETTE['text'],
        linewidth=1.5,
        alpha=0.95,
        zorder=8,
        transform=ax2.transData
    )
    ax2.add_patch(eq_box)
    ax2.text(0.5, 53, r'$T(x) = T_1 - \dfrac{T_1 - T_2}{L}\, x$',
             fontsize=11, ha='center', va='center',
             color=PALETTE['text'], zorder=9)

    # Axis styling
    ax2.set_xlabel(r'Position $x/L$', fontsize=11, labelpad=8,
                   color=PALETTE['text'], fontweight='medium')
    ax2.set_ylabel(r'Temperature $T$ (°C)', fontsize=11, labelpad=8,
                   color=PALETTE['text'], fontweight='medium')

    ax2.set_xlim(-0.12, 1.18)
    ax2.set_ylim(-5, 115)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks([0, 20, 40, 60, 80, 100])

    # Subtle horizontal grid
    ax2.yaxis.grid(True, linestyle='-', alpha=0.5, color='white', linewidth=2)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)

    # Spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['text'])
    ax2.spines['bottom'].set_color(PALETTE['text'])
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.tick_params(colors=PALETTE['text'], width=1.5, labelsize=10)

    # Thermal scale bar
    draw_thermal_scale(ax2, 1.08, 10, 95, T2, T1)

    # Check for overlaps before saving
    fig.canvas.draw()
    print("\nChecking for overlaps...")
    overlaps = check_overlaps(fig, verbose=True)

    if overlaps:
        print(f"\n⚠️  WARNING: {len(overlaps)} overlap(s) detected - please fix!")
    else:
        print("✓ No overlaps - figure is clean")

    plt.savefig('style_experiments/signature_style_final.png', dpi=200,
                facecolor='white', bbox_inches='tight')
    plt.close()

    print("✓ Final signature style created!")


if __name__ == '__main__':
    create_signature_figure()
