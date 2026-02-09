"""
SIGNATURE STYLE for Heat Transfer Textbook
Combines Style 2 aesthetics + Style 4 clarity + distinctive creative elements

Key signature elements:
1. "Thermal glow" - subtle radial gradients suggesting heat emanation
2. Energy flow lines - wavy lines near hot surfaces (heat = molecular motion)
3. Custom arrow style - flame-inspired gradient arrows for heat flow
4. Warm-to-cool gradient that's richer than standard colormaps
5. Soft shadow/depth on key elements
6. Clean annotation boxes with subtle gradient borders
7. A distinctive "thermal scale" sidebar on temperature plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# =============================================================================
# SIGNATURE COLOUR PALETTE
# =============================================================================
# Rich, warm-to-cool gradient inspired by thermal imaging but more elegant
PALETTE = {
    'hot': '#E63946',        # Vibrant warm red
    'warm': '#F4A261',       # Rich amber/orange
    'neutral': '#E9C46A',    # Warm yellow (transition)
    'cool': '#2A9D8F',       # Teal (transition to cold)
    'cold': '#264653',       # Deep blue-grey
    'accent': '#F72585',     # Magenta pink (for special highlights)
    'text': '#1D3557',       # Dark blue-grey for text
    'light_bg': '#F8F9FA',   # Very subtle warm white
    'glow_hot': '#FFE5E5',   # Subtle red glow
    'glow_cold': '#E5F0F8',  # Subtle blue glow
}

# Custom colormap: thermal signature
thermal_colors = [
    (0.0, PALETTE['cold']),
    (0.25, PALETTE['cool']),
    (0.5, PALETTE['neutral']),
    (0.75, PALETTE['warm']),
    (1.0, PALETTE['hot'])
]

def hex_to_rgb(hex_color):
    """Convert hex to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

# Create custom colormap
colors_rgb = [hex_to_rgb(c[1]) for c in thermal_colors]
positions = [c[0] for c in thermal_colors]
THERMAL_CMAP = LinearSegmentedColormap.from_list('thermal_signature',
    list(zip(positions, colors_rgb)))


# =============================================================================
# SIGNATURE VISUAL ELEMENTS
# =============================================================================

def draw_heat_glow(ax, x, y, radius, intensity='hot'):
    """Draw a subtle radial glow around hot elements."""
    glow_color = PALETTE['glow_hot'] if intensity == 'hot' else PALETTE['glow_cold']
    base_color = PALETTE['hot'] if intensity == 'hot' else PALETTE['cold']

    # Multiple concentric circles with decreasing alpha
    for i, alpha in enumerate([0.15, 0.1, 0.05]):
        r = radius * (1 + i * 0.5)
        circle = Circle((x, y), r, facecolor=glow_color,
                        edgecolor='none', alpha=alpha, zorder=0)
        ax.add_patch(circle)


def draw_energy_lines(ax, x_start, y_start, y_end, n_lines=5, side='left'):
    """Draw wavy 'energy/vibration' lines near hot surfaces."""
    direction = -1 if side == 'left' else 1

    for i in range(n_lines):
        y_pos = y_start + (y_end - y_start) * (i + 0.5) / n_lines

        # Create wavy line
        x_wave = np.linspace(0, 0.15, 30)
        y_wave = y_pos + 0.08 * np.sin(x_wave * 40) * (1 - x_wave/0.15)
        x_wave = x_start + direction * x_wave

        # Fade alpha based on distance from surface
        alpha = 0.4 - i * 0.06
        ax.plot(x_wave, y_wave, color=PALETTE['hot'], alpha=alpha,
                linewidth=1.5, solid_capstyle='round')


def draw_heat_arrow(ax, start, end, label=None, label_pos='right'):
    """Draw a distinctive gradient heat flow arrow."""
    x1, y1 = start
    x2, y2 = end

    # Main arrow with gradient effect (using path effects for glow)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=3,
        color=PALETTE['warm'],
        path_effects=[
            pe.Stroke(linewidth=5, foreground=PALETTE['glow_hot'], alpha=0.3),
            pe.Normal()
        ],
        zorder=10
    )
    ax.add_patch(arrow)

    # Label
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        offset = 0.15 if label_pos == 'right' else -0.15
        ax.text(mid_x + offset, mid_y, label, fontsize=14,
                fontweight='bold', color=PALETTE['warm'],
                ha='left' if label_pos == 'right' else 'right',
                va='center')


def draw_annotation_box(ax, x, y, text, width=0.3):
    """Draw a distinctive annotation box with gradient border effect."""
    # Shadow
    shadow = FancyBboxPatch(
        (x - width/2 + 0.02, y - 0.08),
        width, 0.16,
        boxstyle='round,pad=0.02,rounding_size=0.02',
        facecolor='#00000010',
        edgecolor='none',
        zorder=4
    )
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch(
        (x - width/2, y - 0.06),
        width, 0.16,
        boxstyle='round,pad=0.02,rounding_size=0.02',
        facecolor='white',
        edgecolor=PALETTE['text'],
        linewidth=1.5,
        zorder=5
    )
    ax.add_patch(box)

    ax.text(x, y + 0.02, text, fontsize=11, ha='center', va='center',
            color=PALETTE['text'], zorder=6)


def draw_thermal_scale(ax, x, y_bottom, height, t_min, t_max):
    """Draw a distinctive thermal scale bar (like a thermometer)."""
    width = 0.06
    n_segments = 50

    # Draw gradient bar
    for i in range(n_segments):
        y0 = y_bottom + i * height / n_segments
        y1 = y_bottom + (i + 1) * height / n_segments
        frac = i / n_segments
        color = THERMAL_CMAP(frac)
        ax.fill([x, x, x + width, x + width], [y0, y1, y1, y0],
                color=color, edgecolor='none')

    # Border with rounded ends
    ax.plot([x, x], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.5)
    ax.plot([x + width, x + width], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.5)

    # Tick marks and labels
    for frac, label in [(0, f'{t_min}°C'), (0.5, f'{(t_min+t_max)/2:.0f}°C'), (1, f'{t_max}°C')]:
        y = y_bottom + frac * height
        ax.plot([x + width, x + width + 0.02], [y, y],
                color=PALETTE['text'], linewidth=1)
        ax.text(x + width + 0.04, y, label, fontsize=9,
                va='center', color=PALETTE['text'])


# =============================================================================
# SIGNATURE STYLE FIGURE
# =============================================================================

def create_signature_figure():
    """Create the signature style figure for the heat transfer textbook."""

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Custom layout
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])   # Physical diagram
    ax2 = fig.add_axes([0.52, 0.1, 0.42, 0.8])  # Temperature plot

    # === LEFT: Physical System ===
    ax1.set_xlim(-0.4, 1.8)
    ax1.set_ylim(-0.7, 5.0)
    ax1.set_facecolor('white')

    # Draw thermal glow behind hot surface
    draw_heat_glow(ax1, 0.65, 4.0, 0.5, 'hot')

    # Draw energy/vibration lines near hot surface
    draw_energy_lines(ax1, 0.25, 3.5, 4.3, n_lines=4, side='left')

    # Draw the slab with rich thermal gradient
    n_strips = 60
    slab_bottom, slab_top = 0.2, 4.0
    slab_left, slab_right = 0.25, 1.05

    for i in range(n_strips):
        y0 = slab_bottom + i * (slab_top - slab_bottom) / n_strips
        y1 = slab_bottom + (i + 1) * (slab_top - slab_bottom) / n_strips
        frac = 1 - i / n_strips  # Hot at top
        color = THERMAL_CMAP(frac)
        ax1.fill([slab_left, slab_left, slab_right, slab_right],
                 [y0, y1, y1, y0], color=color, edgecolor='none')

    # Slab border with subtle shadow
    shadow_offset = 0.03
    ax1.plot([slab_left + shadow_offset, slab_left + shadow_offset,
              slab_right + shadow_offset, slab_right + shadow_offset,
              slab_left + shadow_offset],
             [slab_bottom - shadow_offset, slab_top - shadow_offset,
              slab_top - shadow_offset, slab_bottom - shadow_offset,
              slab_bottom - shadow_offset],
             color='#00000020', linewidth=6, solid_capstyle='round')

    ax1.plot([slab_left, slab_left, slab_right, slab_right, slab_left],
             [slab_bottom, slab_top, slab_top, slab_bottom, slab_bottom],
             color=PALETTE['text'], linewidth=2.5)

    # Temperature labels with distinctive styling
    # Hot label (top)
    ax1.text(0.65, 4.45, r'$T_1$', fontsize=16, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['hot'],
                      edgecolor='none', alpha=0.95),
             zorder=10)
    ax1.text(0.65, 4.75, 'hot surface', fontsize=10, ha='center',
             color=PALETTE['hot'], style='italic')

    # Cold label (bottom)
    ax1.text(0.65, -0.2, r'$T_2$', fontsize=16, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['cold'],
                      edgecolor='none', alpha=0.95),
             zorder=10)
    ax1.text(0.65, -0.5, 'cold surface', fontsize=10, ha='center',
             color=PALETTE['cold'], style='italic')

    # Heat flow arrow
    draw_heat_arrow(ax1, (1.4, 3.6), (1.4, 0.6), r'$\dot{Q}$', 'right')

    # Dimension annotation
    ax1.annotate('', xy=(slab_right, -0.05), xytext=(slab_left, -0.05),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.5,
                               color=PALETTE['text'], mutation_scale=10))
    ax1.text(0.65, -0.18, r'$L$', fontsize=13, ha='center',
             color=PALETTE['text'], fontweight='medium')

    # Material property label
    ax1.text(0.08, 2.1, 'Thermal\nconductivity', fontsize=9, ha='right',
             va='center', color=PALETTE['text'], linespacing=1.3)
    ax1.text(0.08, 1.6, r'$k$', fontsize=16, ha='right', va='center',
             color=PALETTE['text'], fontweight='bold', style='italic')

    ax1.axis('off')

    # === RIGHT: Temperature Profile ===
    ax2.set_facecolor(PALETTE['light_bg'])

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x

    # Gradient fill under curve using our thermal colormap
    for i in range(len(x) - 1):
        frac = 1 - x[i]  # Hot at x=0
        color = THERMAL_CMAP(frac)
        ax2.fill_between(x[i:i+2], 0, T[i:i+2], color=color, alpha=0.35)

    # Main line with glow effect
    ax2.plot(x, T, color=PALETTE['text'], linewidth=3,
             path_effects=[
                 pe.Stroke(linewidth=6, foreground='white', alpha=0.8),
                 pe.Normal()
             ], zorder=5)

    # Endpoint markers with glow
    ax2.scatter([0], [T1], s=120, color=PALETTE['hot'], zorder=10,
                edgecolors='white', linewidths=2.5)
    ax2.scatter([1], [T2], s=120, color=PALETTE['cold'], zorder=10,
                edgecolors='white', linewidths=2.5)

    # Direct labels
    ax2.annotate(r'$T_1 = 100°C$', xy=(0, T1), xytext=(-0.08, T1),
                fontsize=11, ha='right', va='center', color=PALETTE['hot'],
                fontweight='semibold')
    ax2.annotate(r'$T_2 = 20°C$', xy=(1, T2), xytext=(1.08, T2),
                fontsize=11, ha='left', va='center', color=PALETTE['cold'],
                fontweight='semibold')

    # Equation box with signature styling
    eq_box = FancyBboxPatch(
        (0.28, 38), 0.44, 24,
        boxstyle='round,pad=0.02,rounding_size=0.03',
        facecolor='white',
        edgecolor=PALETTE['text'],
        linewidth=1.5,
        alpha=0.95,
        zorder=8,
        transform=ax2.transData
    )
    ax2.add_patch(eq_box)
    ax2.text(0.5, 50, r'$T(x) = T_1 - \dfrac{T_1 - T_2}{L}\, x$',
             fontsize=12, ha='center', va='center',
             color=PALETTE['text'], zorder=9)

    # Axis styling
    ax2.set_xlabel(r'Position $x/L$', fontsize=12, labelpad=10,
                   color=PALETTE['text'], fontweight='medium')
    ax2.set_ylabel(r'Temperature $T$ (°C)', fontsize=12, labelpad=10,
                   color=PALETTE['text'], fontweight='medium')

    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(-5, 115)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Subtle grid - horizontal only
    ax2.yaxis.grid(True, linestyle='-', alpha=0.5, color='white', linewidth=2)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)

    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['text'])
    ax2.spines['bottom'].set_color(PALETTE['text'])
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.tick_params(colors=PALETTE['text'], width=1.5, labelsize=10)

    # Add thermal scale bar (signature element)
    draw_thermal_scale(ax2, 1.05, 10, 90, T2, T1)

    plt.savefig('style_experiments/signature_style.png', dpi=200,
                facecolor='white', bbox_inches='tight')
    plt.close()

    print("✓ Signature style created!")


# =============================================================================
# ALTERNATIVE: Even more whimsical version
# =============================================================================

def create_signature_v2():
    """Version 2: More playful with additional distinctive elements."""

    fig = plt.figure(figsize=(12, 5.5))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_axes([0.05, 0.12, 0.4, 0.78])
    ax2 = fig.add_axes([0.52, 0.12, 0.42, 0.78])

    # === LEFT: Physical System with more flair ===
    ax1.set_xlim(-0.5, 1.9)
    ax1.set_ylim(-0.8, 5.2)
    ax1.set_facecolor('white')

    # Multiple glow layers for more dramatic effect
    for r, alpha in [(0.8, 0.08), (0.6, 0.12), (0.4, 0.15)]:
        circle = Circle((0.65, 4.2), r, facecolor=PALETTE['glow_hot'],
                        edgecolor='none', alpha=alpha, zorder=0)
        ax1.add_patch(circle)

    # More prominent energy lines
    for i in range(6):
        y_base = 3.3 + i * 0.25
        x_wave = np.linspace(0, 0.25, 40)
        # Varying wave frequencies for visual interest
        freq = 35 + i * 5
        y_wave = y_base + 0.06 * np.sin(x_wave * freq) * np.exp(-x_wave * 4)
        x_wave = 0.2 - x_wave
        alpha = 0.5 - i * 0.07
        ax1.plot(x_wave, y_wave, color=PALETTE['hot'], alpha=alpha,
                linewidth=2, solid_capstyle='round')

    # Slab with enhanced gradient
    n_strips = 80
    slab_bottom, slab_top = 0.2, 4.0
    slab_left, slab_right = 0.25, 1.05

    for i in range(n_strips):
        y0 = slab_bottom + i * (slab_top - slab_bottom) / n_strips
        y1 = slab_bottom + (i + 1) * (slab_top - slab_bottom) / n_strips
        frac = 1 - i / n_strips
        color = THERMAL_CMAP(frac)
        ax1.fill([slab_left, slab_left, slab_right, slab_right],
                 [y0, y1, y1, y0], color=color, edgecolor='none')

    # Border
    ax1.plot([slab_left, slab_left, slab_right, slab_right, slab_left],
             [slab_bottom, slab_top, slab_top, slab_bottom, slab_bottom],
             color=PALETTE['text'], linewidth=2.5)

    # Labels with icons/symbols
    # Hot: flame-inspired decoration
    ax1.text(0.65, 4.5, '🔥', fontsize=14, ha='center', va='center', zorder=9)
    ax1.text(0.65, 4.25, r'$T_1$', fontsize=18, ha='center', va='center',
             color=PALETTE['hot'], fontweight='bold')

    # Cold: snowflake-inspired
    ax1.text(0.65, -0.25, '❄', fontsize=12, ha='center', va='center', zorder=9)
    ax1.text(0.65, -0.5, r'$T_2$', fontsize=18, ha='center', va='center',
             color=PALETTE['cold'], fontweight='bold')

    # Fancy heat flow arrow with multiple elements
    # Outer glow
    for offset, alpha, lw in [(0, 0.15, 8), (0, 0.25, 5)]:
        ax1.annotate('', xy=(1.5, 0.5), xytext=(1.5, 3.5),
                    arrowprops=dict(arrowstyle='->', lw=lw,
                                   color=PALETTE['warm'], alpha=alpha,
                                   mutation_scale=25))
    # Main arrow
    ax1.annotate('', xy=(1.5, 0.5), xytext=(1.5, 3.5),
                arrowprops=dict(arrowstyle='-|>', lw=3,
                               color=PALETTE['warm'], mutation_scale=22))

    # Animated-looking heat flow label
    ax1.text(1.65, 2.0, r'$\dot{Q}$', fontsize=16, ha='left', va='center',
             color=PALETTE['warm'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=PALETTE['warm'], linewidth=1.5, alpha=0.9))

    # Dimension
    ax1.annotate('', xy=(slab_right, 0.0), xytext=(slab_left, 0.0),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.5,
                               color=PALETTE['text'], mutation_scale=10))
    ax1.text(0.65, -0.12, r'thickness $L$', fontsize=10, ha='center',
             color=PALETTE['text'])

    # Material property with decorative box
    prop_box = FancyBboxPatch(
        (-0.15, 1.7), 0.35, 0.7,
        boxstyle='round,pad=0.02,rounding_size=0.05',
        facecolor='white',
        edgecolor=PALETTE['cool'],
        linewidth=1.5,
        alpha=0.9
    )
    ax1.add_patch(prop_box)
    ax1.text(0.02, 2.15, 'conductivity', fontsize=9, ha='center',
             color=PALETTE['text'])
    ax1.text(0.02, 1.9, r'$k$', fontsize=18, ha='center',
             color=PALETTE['cool'], fontweight='bold', style='italic')

    ax1.axis('off')

    # === RIGHT: Temperature Profile ===
    ax2.set_facecolor('#FAFBFC')

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x

    # Rich gradient fill
    for i in range(len(x) - 1):
        frac = 1 - x[i]
        color = THERMAL_CMAP(frac)
        ax2.fill_between(x[i:i+2], 0, T[i:i+2], color=color, alpha=0.4)

    # Main line with shadow
    ax2.plot(x, T - 2, color='#00000015', linewidth=5, solid_capstyle='round')
    ax2.plot(x, T, color=PALETTE['text'], linewidth=3.5, solid_capstyle='round')

    # Fancy endpoint markers
    for pos, temp, color in [(0, T1, PALETTE['hot']), (1, T2, PALETTE['cold'])]:
        # Glow
        ax2.scatter([pos], [temp], s=300, color=color, alpha=0.2, zorder=8)
        # Main marker
        ax2.scatter([pos], [temp], s=140, color=color, zorder=10,
                    edgecolors='white', linewidths=3)

    # Labels with connecting lines
    ax2.plot([-0.02, -0.08], [T1, T1], color=PALETTE['hot'], linewidth=1.5)
    ax2.text(-0.1, T1, r'$T_1$', fontsize=13, ha='right', va='center',
             color=PALETTE['hot'], fontweight='bold')

    ax2.plot([1.02, 1.08], [T2, T2], color=PALETTE['cold'], linewidth=1.5)
    ax2.text(1.1, T2, r'$T_2$', fontsize=13, ha='left', va='center',
             color=PALETTE['cold'], fontweight='bold')

    # Equation in styled box
    eq_y = 52
    # Shadow
    eq_box_shadow = FancyBboxPatch(
        (0.18, eq_y - 14), 0.64, 28,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        facecolor='#00000008',
        edgecolor='none',
        transform=ax2.transData
    )
    ax2.add_patch(eq_box_shadow)

    # Box with gradient-like border (simulated with multiple boxes)
    eq_box = FancyBboxPatch(
        (0.16, eq_y - 12), 0.64, 28,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        facecolor='white',
        edgecolor=PALETTE['text'],
        linewidth=2,
        transform=ax2.transData,
        zorder=8
    )
    ax2.add_patch(eq_box)

    ax2.text(0.48, eq_y + 2, r'$T(x) = T_1 - \dfrac{T_1 - T_2}{L}\, x$',
             fontsize=13, ha='center', va='center',
             color=PALETTE['text'], zorder=9)

    # "Linear profile" label with flair
    ax2.text(0.48, eq_y - 8, 'linear temperature profile',
             fontsize=9, ha='center', va='center',
             color=PALETTE['cool'], style='italic', zorder=9)

    # Axis styling
    ax2.set_xlabel(r'Position $x/L$', fontsize=12, labelpad=10,
                   color=PALETTE['text'], fontweight='medium')
    ax2.set_ylabel(r'Temperature $T$ (°C)', fontsize=12, labelpad=10,
                   color=PALETTE['text'], fontweight='medium')

    ax2.set_xlim(-0.18, 1.18)
    ax2.set_ylim(-8, 118)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Grid
    ax2.yaxis.grid(True, linestyle='-', alpha=0.6, color='white', linewidth=2)
    ax2.set_axisbelow(True)

    # Spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['text'])
    ax2.spines['bottom'].set_color(PALETTE['text'])
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.tick_params(colors=PALETTE['text'], width=1.5, labelsize=10)

    plt.savefig('style_experiments/signature_style_v2.png', dpi=200,
                facecolor='white', bbox_inches='tight')
    plt.close()

    print("✓ Signature style v2 (more whimsical) created!")


# =============================================================================
# Run
# =============================================================================
if __name__ == '__main__':
    create_signature_figure()
    create_signature_v2()
    print("\nSignature styles saved to style_experiments/")
