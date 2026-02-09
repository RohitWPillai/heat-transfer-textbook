"""
Compare different figure styles for the heat transfer textbook.
Run this script to generate comparison images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Create output directory
import os
os.makedirs('style_experiments', exist_ok=True)

# =============================================================================
# STYLE 1: Clean Minimalist (Tufte-inspired)
# =============================================================================
def style1_minimalist():
    """Clean, minimal style with focus on data, not decoration."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('white')

    # Colour palette - muted, sophisticated
    HOT = '#c0392b'
    COLD = '#2980b9'
    NEUTRAL = '#2c3e50'
    LIGHT_FILL = '#ecf0f1'

    # === Left: Physical diagram with T1 on top ===
    ax1.set_xlim(-0.3, 1.8)
    ax1.set_ylim(-0.5, 4.5)

    # Draw slab as simple rectangle with gradient
    n_strips = 40
    for i in range(n_strips):
        y_bot = i * 3.5 / n_strips
        y_top = (i + 1) * 3.5 / n_strips
        # Gradient from hot (top) to cold (bottom)
        frac = 1 - i / n_strips
        color = plt.cm.RdYlBu_r(frac * 0.8 + 0.1)
        ax1.fill([0.3, 0.3, 1.0, 1.0], [y_bot, y_top, y_top, y_bot],
                 color=color, edgecolor='none')

    # Clean border
    ax1.plot([0.3, 0.3, 1.0, 1.0, 0.3], [0, 3.5, 3.5, 0, 0],
             color=NEUTRAL, linewidth=1.5)

    # Temperature labels - T1 at top, T2 at bottom
    ax1.text(0.65, 3.8, r'$T_1$ (hot)', fontsize=11, ha='center',
             color=HOT, fontweight='medium')
    ax1.text(0.65, -0.3, r'$T_2$ (cold)', fontsize=11, ha='center',
             color=COLD, fontweight='medium')

    # Heat flow arrow (downward)
    ax1.annotate('', xy=(1.4, 0.5), xytext=(1.4, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color=HOT,
                               mutation_scale=15))
    ax1.text(1.55, 1.75, r'$\dot{Q}$', fontsize=12, ha='left',
             va='center', color=HOT, fontweight='bold')

    # Thickness dimension
    ax1.annotate('', xy=(1.0, -0.15), xytext=(0.3, -0.15),
                arrowprops=dict(arrowstyle='<->', lw=1, color=NEUTRAL))
    ax1.text(0.65, -0.35, r'$L$', fontsize=11, ha='center', color=NEUTRAL)

    # Properties label
    ax1.text(0.65, 1.75, r'$k$', fontsize=14, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='circle,pad=0.3', facecolor=NEUTRAL, alpha=0.7))

    ax1.axis('off')
    ax1.set_title('Physical System', fontsize=12, fontweight='bold',
                  color=NEUTRAL, pad=10)

    # === Right: Temperature profile ===
    ax2.set_facecolor('white')

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x  # Linear decrease

    # Main line
    ax2.plot(x, T, color=NEUTRAL, linewidth=2.5)

    # Subtle fill
    ax2.fill_between(x, T2, T, alpha=0.15, color=COLD)

    # Reference lines (subtle)
    ax2.axhline(y=T1, color=HOT, linestyle=':', alpha=0.6, linewidth=1)
    ax2.axhline(y=T2, color=COLD, linestyle=':', alpha=0.6, linewidth=1)

    # Direct labels (no legend needed)
    ax2.text(1.02, T1, r'$T_1$', fontsize=11, va='center', color=HOT)
    ax2.text(1.02, T2, r'$T_2$', fontsize=11, va='center', color=COLD)

    # Slope annotation - simple, not boxed
    ax2.text(0.5, 65, r'slope $= \dfrac{T_2 - T_1}{L}$', fontsize=10,
             ha='center', color=NEUTRAL, alpha=0.8)

    ax2.set_xlabel(r'Position $x/L$', fontsize=11)
    ax2.set_ylabel(r'Temperature $T$ (°C)', fontsize=11)
    ax2.set_title('Temperature Profile', fontsize=12, fontweight='bold',
                  color=NEUTRAL, pad=10)
    ax2.set_xlim(0, 1.15)
    ax2.set_ylim(0, 115)

    # Minimal spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(NEUTRAL)
    ax2.spines['bottom'].set_color(NEUTRAL)
    ax2.tick_params(colors=NEUTRAL)

    # No grid - Tufte style
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig('style_experiments/style1_minimalist.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    plt.close()


# =============================================================================
# STYLE 2: Modern Editorial (inspired by The Economist, FiveThirtyEight)
# =============================================================================
def style2_editorial():
    """Bold, confident style with strong visual hierarchy."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#fafafa')

    # Colour palette - bolder, more contrast
    HOT = '#e74c3c'
    COLD = '#3498db'
    NEUTRAL = '#1a1a2e'
    ACCENT = '#f39c12'
    BG = '#fafafa'

    # === Left: Physical diagram ===
    ax1.set_facecolor(BG)
    ax1.set_xlim(-0.3, 1.8)
    ax1.set_ylim(-0.5, 4.5)

    # Slab with bold gradient
    n_strips = 50
    for i in range(n_strips):
        y_bot = i * 3.5 / n_strips
        y_top = (i + 1) * 3.5 / n_strips
        frac = 1 - i / n_strips
        color = plt.cm.coolwarm(frac)
        ax1.fill([0.3, 0.3, 1.0, 1.0], [y_bot, y_top, y_top, y_bot],
                 color=color, edgecolor='none')

    # Bold border
    ax1.plot([0.3, 0.3, 1.0, 1.0, 0.3], [0, 3.5, 3.5, 0, 0],
             color=NEUTRAL, linewidth=2.5)

    # Temperature labels with background pills
    ax1.text(0.65, 3.9, r'$T_1$', fontsize=13, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=HOT, edgecolor='none'))
    ax1.text(0.65, -0.4, r'$T_2$', fontsize=13, ha='center', va='center',
             color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLD, edgecolor='none'))

    # Heat flow - thick arrow
    ax1.annotate('', xy=(1.5, 0.3), xytext=(1.5, 3.2),
                arrowprops=dict(arrowstyle='->', lw=3, color=ACCENT,
                               mutation_scale=20))
    ax1.text(1.65, 1.75, r'$\dot{Q}$', fontsize=14, ha='left',
             va='center', color=ACCENT, fontweight='bold')

    # Dimension
    ax1.annotate('', xy=(1.0, -0.2), xytext=(0.3, -0.2),
                arrowprops=dict(arrowstyle='|-|', lw=1.5, color=NEUTRAL))
    ax1.text(0.65, -0.4, r'$L$', fontsize=12, ha='center', color=NEUTRAL,
             fontweight='bold')

    ax1.axis('off')
    ax1.set_title('Physical System', fontsize=13, fontweight='bold',
                  color=NEUTRAL, pad=15)

    # === Right: Temperature profile ===
    ax2.set_facecolor(BG)

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x

    # Gradient fill under curve
    for i in range(len(x)-1):
        frac = 1 - x[i]
        color = plt.cm.coolwarm(frac)
        ax2.fill_between(x[i:i+2], T2, T[i:i+2], color=color, alpha=0.4)

    # Bold line
    ax2.plot(x, T, color=NEUTRAL, linewidth=3)

    # Data points at ends
    ax2.scatter([0, 1], [T1, T2], s=80, zorder=5,
                c=[HOT, COLD], edgecolors='white', linewidths=2)

    # Labels
    ax2.text(-0.08, T1, r'$T_1$', fontsize=12, va='center', ha='right',
             color=HOT, fontweight='bold')
    ax2.text(1.08, T2, r'$T_2$', fontsize=12, va='center', ha='left',
             color=COLD, fontweight='bold')

    ax2.set_xlabel(r'Position $x/L$', fontsize=12, fontweight='medium')
    ax2.set_ylabel(r'Temperature (°C)', fontsize=12, fontweight='medium')
    ax2.set_title('Linear Temperature Profile', fontsize=13, fontweight='bold',
                  color=NEUTRAL, pad=15)
    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(0, 115)

    # Grid - horizontal only
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color=NEUTRAL)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)

    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('style_experiments/style2_editorial.png', dpi=150,
                facecolor=BG, bbox_inches='tight')
    plt.close()


# =============================================================================
# STYLE 3: Academic/SciencePlots inspired
# =============================================================================
def style3_academic():
    """Publication-quality, clean academic style."""

    # Try to use science style if available
    try:
        plt.style.use(['science', 'no-latex'])
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    # Colour palette - Paul Tol inspired, colorblind safe
    HOT = '#CC3311'  # Red
    COLD = '#0077BB'  # Blue
    NEUTRAL = '#333333'
    GREY = '#BBBBBB'

    # === Left: Schematic diagram ===
    ax1.set_xlim(-0.2, 1.6)
    ax1.set_ylim(-0.4, 4.2)

    # Simple slab - just outlines with hatching for material
    rect = plt.Rectangle((0.3, 0), 0.7, 3.5, fill=True,
                          facecolor='#f0f0f0', edgecolor=NEUTRAL,
                          linewidth=1.5, hatch='///')
    ax1.add_patch(rect)

    # Temperature indicators - simple lines at boundaries
    ax1.plot([0.1, 0.3], [3.5, 3.5], color=HOT, linewidth=3)
    ax1.plot([0.1, 0.3], [0, 0], color=COLD, linewidth=3)

    # Labels
    ax1.text(0.05, 3.5, r'$T_1$', fontsize=11, va='center', ha='right', color=HOT)
    ax1.text(0.05, 0, r'$T_2$', fontsize=11, va='center', ha='right', color=COLD)

    # Heat flow
    ax1.annotate('', xy=(1.2, 0.5), xytext=(1.2, 3.0),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color=NEUTRAL))
    ax1.text(1.3, 1.75, r'$\dot{Q}$', fontsize=11, ha='left', va='center')

    # Dimension
    ax1.plot([0.3, 0.3], [-0.15, -0.05], color=NEUTRAL, linewidth=1)
    ax1.plot([1.0, 1.0], [-0.15, -0.05], color=NEUTRAL, linewidth=1)
    ax1.annotate('', xy=(1.0, -0.1), xytext=(0.3, -0.1),
                arrowprops=dict(arrowstyle='<->', lw=1, color=NEUTRAL))
    ax1.text(0.65, -0.25, r'$L$', fontsize=11, ha='center')

    # Material property
    ax1.text(0.65, 1.75, r'$k$', fontsize=12, ha='center', va='center',
             style='italic')

    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title('(a) Conduction through slab', fontsize=11,
                  loc='left', pad=10)

    # === Right: Temperature profile ===
    ax2.set_facecolor('white')

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x

    ax2.plot(x, T, color=NEUTRAL, linewidth=1.5, solid_capstyle='round')

    # Markers at boundaries
    ax2.plot(0, T1, 'o', color=HOT, markersize=6)
    ax2.plot(1, T2, 'o', color=COLD, markersize=6)

    # Dashed lines to axes
    ax2.plot([0, 0], [0, T1], ':', color=GREY, linewidth=0.8)
    ax2.plot([1, 1], [0, T2], ':', color=GREY, linewidth=0.8)

    ax2.set_xlabel(r'$x/L$', fontsize=11)
    ax2.set_ylabel(r'$T$ (°C)', fontsize=11)
    ax2.set_title('(b) Temperature distribution', fontsize=11,
                  loc='left', pad=10)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 110)

    # Tick styling
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks([0, 20, 40, 60, 80, 100])

    # Subtle grid
    ax2.grid(True, linestyle='-', alpha=0.2, linewidth=0.5)

    # Clean spines
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('style_experiments/style3_academic.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    plt.close()

    # Reset style
    plt.style.use('default')


# =============================================================================
# STYLE 4: Modern Textbook (clean, approachable, colourful but not garish)
# =============================================================================
def style4_modern_textbook():
    """Balance of visual appeal and clarity - recommended for textbooks."""

    fig = plt.figure(figsize=(11, 4.5))
    fig.patch.set_facecolor('white')

    # Create custom gridspec for better control
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Colour palette - warm and approachable
    HOT = '#E63946'      # Warm red
    WARM = '#F4A261'     # Orange
    COLD = '#457B9D'     # Steel blue
    NEUTRAL = '#1D3557'  # Dark blue-grey
    LIGHT = '#A8DADC'    # Light cyan
    BG_ACCENT = '#F1FAEE'  # Very light green-white

    # === Left: Physical diagram ===
    ax1.set_xlim(-0.3, 1.7)
    ax1.set_ylim(-0.6, 4.8)
    ax1.set_facecolor('white')

    # Draw slab with smooth gradient (vertical - hot at top)
    n = 60
    for i in range(n):
        y0 = i * 3.8 / n
        y1 = (i + 1) * 3.8 / n
        # Custom gradient: red -> orange -> light blue -> blue
        t = i / n
        if t < 0.5:
            # Hot to warm
            r = 0.9 - 0.2 * (t * 2)
            g = 0.22 + 0.4 * (t * 2)
            b = 0.28 + 0.1 * (t * 2)
        else:
            # Warm to cold
            t2 = (t - 0.5) * 2
            r = 0.7 - 0.43 * t2
            g = 0.62 - 0.14 * t2
            b = 0.38 + 0.23 * t2
        ax1.fill([0.3, 0.3, 1.0, 1.0], [y0, y1, y1, y0],
                 color=(r, g, b), edgecolor='none')

    # Border with shadow effect
    shadow = plt.Rectangle((0.32, -0.02), 0.7, 3.8,
                           facecolor='none', edgecolor='#cccccc',
                           linewidth=4, zorder=0)
    ax1.add_patch(shadow)
    ax1.plot([0.3, 0.3, 1.0, 1.0, 0.3], [0, 3.8, 3.8, 0, 0],
             color=NEUTRAL, linewidth=2, zorder=2)

    # Temperature labels
    ax1.text(0.65, 4.2, r'$T_1$ = hot surface', fontsize=11, ha='center',
             color=HOT, fontweight='semibold')
    ax1.text(0.65, -0.35, r'$T_2$ = cold surface', fontsize=11, ha='center',
             color=COLD, fontweight='semibold')

    # Heat flow arrow with label
    arrow_x = 1.35
    ax1.annotate('', xy=(arrow_x, 0.4), xytext=(arrow_x, 3.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=WARM,
                               connectionstyle='arc3', mutation_scale=18))
    ax1.text(arrow_x + 0.15, 1.9, 'Heat\nflow', fontsize=9, ha='left',
             va='center', color=NEUTRAL, linespacing=1.2)
    ax1.text(arrow_x + 0.15, 1.9 - 0.6, r'$\dot{Q}$', fontsize=13, ha='left',
             va='top', color=WARM, fontweight='bold')

    # Thickness annotation
    ax1.annotate('', xy=(1.0, -0.15), xytext=(0.3, -0.15),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.2, color=NEUTRAL,
                               mutation_scale=8))
    ax1.text(0.65, -0.5, r'thickness $L$', fontsize=10, ha='center',
             color=NEUTRAL)

    # Material label
    ax1.text(0.15, 1.9, 'Material\n' + r'($k$)', fontsize=9, ha='right',
             va='center', color=NEUTRAL, linespacing=1.3)

    ax1.axis('off')

    # === Right: Temperature profile ===
    ax2.set_facecolor(BG_ACCENT)

    x = np.linspace(0, 1, 100)
    T1, T2 = 100, 20
    T = T1 - (T1 - T2) * x

    # Gradient fill matching the slab colours
    for i in range(len(x)-1):
        t = x[i]
        if t < 0.5:
            r = 0.9 - 0.2 * (t * 2)
            g = 0.22 + 0.4 * (t * 2)
            b = 0.28 + 0.1 * (t * 2)
        else:
            t2 = (t - 0.5) * 2
            r = 0.7 - 0.43 * t2
            g = 0.62 - 0.14 * t2
            b = 0.38 + 0.23 * t2
        ax2.fill_between(x[i:i+2], 0, T[i:i+2], color=(r, g, b), alpha=0.25)

    # Main line - bold
    ax2.plot(x, T, color=NEUTRAL, linewidth=2.5, solid_capstyle='round')

    # End markers
    ax2.scatter([0], [T1], s=100, color=HOT, zorder=5, edgecolors='white', linewidths=2)
    ax2.scatter([1], [T2], s=100, color=COLD, zorder=5, edgecolors='white', linewidths=2)

    # Direct labels
    ax2.annotate(r'$T_1 = 100°C$', xy=(0, T1), xytext=(-0.12, T1),
                fontsize=10, ha='right', va='center', color=HOT,
                fontweight='medium')
    ax2.annotate(r'$T_2 = 20°C$', xy=(1, T2), xytext=(1.12, T2),
                fontsize=10, ha='left', va='center', color=COLD,
                fontweight='medium')

    # Equation in a nice box
    eq_text = r'$T(x) = T_1 - \frac{T_1 - T_2}{L}x$'
    ax2.text(0.5, 45, eq_text, fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=NEUTRAL, alpha=0.9, linewidth=1))

    ax2.set_xlabel(r'Position $x/L$', fontsize=11, labelpad=8)
    ax2.set_ylabel(r'Temperature $T$ (°C)', fontsize=11, labelpad=8)

    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-5, 115)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Subtle horizontal grid only
    ax2.yaxis.grid(True, linestyle='-', alpha=0.4, color='white', linewidth=1.5)
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)

    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(NEUTRAL)
    ax2.spines['bottom'].set_color(NEUTRAL)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    ax2.tick_params(colors=NEUTRAL, width=1.2)

    plt.tight_layout()
    plt.savefig('style_experiments/style4_modern_textbook.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    plt.close()


# =============================================================================
# Run all styles
# =============================================================================
if __name__ == '__main__':
    print("Generating style comparisons...")

    style1_minimalist()
    print("✓ Style 1: Minimalist (Tufte-inspired)")

    style2_editorial()
    print("✓ Style 2: Editorial (FiveThirtyEight-inspired)")

    style3_academic()
    print("✓ Style 3: Academic (SciencePlots-inspired)")

    style4_modern_textbook()
    print("✓ Style 4: Modern Textbook (recommended)")

    print("\nAll styles saved to style_experiments/")
    print("Open the PNG files to compare.")
