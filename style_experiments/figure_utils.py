"""
Utility functions for the Heat Transfer Textbook figures.
Includes automatic overlap detection and the signature style elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings


# =============================================================================
# OVERLAP DETECTION SYSTEM
# =============================================================================

def get_text_bbox(text_obj, renderer):
    """Get the bounding box of a text object in data coordinates."""
    bbox = text_obj.get_window_extent(renderer=renderer)
    ax = text_obj.axes
    bbox_data = bbox.transformed(ax.transData.inverted())
    return bbox_data


def get_element_bboxes(ax, renderer):
    """
    Extract bounding boxes for all text elements and patches in an axes.
    Returns list of (bbox, label, element_type) tuples.
    """
    bboxes = []

    # Get text elements (use bbox patch extent if present, to include pill padding)
    for i, text in enumerate(ax.texts):
        if text.get_text().strip():  # Skip empty text
            try:
                bp = text.get_bbox_patch()
                if bp is not None:
                    bbox = bp.get_window_extent(renderer).transformed(ax.transData.inverted())
                else:
                    bbox = get_text_bbox(text, renderer)
                label = f"Text: '{text.get_text()[:20]}...'" if len(text.get_text()) > 20 else f"Text: '{text.get_text()}'"
                bboxes.append((bbox, label, 'text', text))
            except:
                pass

    # Get axis labels
    if ax.xaxis.label.get_text():
        try:
            bbox = get_text_bbox(ax.xaxis.label, renderer)
            bboxes.append((bbox, 'X-axis label', 'axis_label', ax.xaxis.label))
        except:
            pass

    if ax.yaxis.label.get_text():
        try:
            bbox = get_text_bbox(ax.yaxis.label, renderer)
            bboxes.append((bbox, 'Y-axis label', 'axis_label', ax.yaxis.label))
        except:
            pass

    # Get tick labels
    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text():
            try:
                bbox = get_text_bbox(tick.label1, renderer)
                bboxes.append((bbox, f"X-tick: '{tick.label1.get_text()}'", 'tick', tick.label1))
            except:
                pass

    for tick in ax.yaxis.get_major_ticks():
        if tick.label1.get_text():
            try:
                bbox = get_text_bbox(tick.label1, renderer)
                bboxes.append((bbox, f"Y-tick: '{tick.label1.get_text()}'", 'tick', tick.label1))
            except:
                pass

    return bboxes


def boxes_overlap(bbox1, bbox2, padding=0.02):
    """
    Check if two bounding boxes overlap.
    padding: relative padding to add around boxes (fraction of box size)
    """
    # Add padding
    w1 = bbox1.width * padding
    h1 = bbox1.height * padding
    w2 = bbox2.width * padding
    h2 = bbox2.height * padding

    # Check overlap
    return not (bbox1.x1 + w1 < bbox2.x0 - w2 or  # bbox1 is left of bbox2
                bbox1.x0 - w1 > bbox2.x1 + w2 or  # bbox1 is right of bbox2
                bbox1.y1 + h1 < bbox2.y0 - h2 or  # bbox1 is below bbox2
                bbox1.y0 - h1 > bbox2.y1 + h2)    # bbox1 is above bbox2


def check_overlaps(fig, verbose=True):
    """
    Check all axes in a figure for overlapping text elements.

    Returns:
        list of tuples: (ax_index, element1_label, element2_label) for each overlap

    Usage:
        fig, ax = plt.subplots()
        # ... create figure ...
        overlaps = check_overlaps(fig)
        if overlaps:
            print("WARNING: Overlapping elements detected!")
    """
    renderer = fig.canvas.get_renderer()
    all_overlaps = []

    for ax_idx, ax in enumerate(fig.axes):
        bboxes = get_element_bboxes(ax, renderer)

        # Check all pairs
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                bbox1, label1, type1, elem1 = bboxes[i]
                bbox2, label2, type2, elem2 = bboxes[j]

                # Skip tick-to-tick comparisons (they're meant to be close)
                if type1 == 'tick' and type2 == 'tick':
                    continue

                if boxes_overlap(bbox1, bbox2):
                    all_overlaps.append((ax_idx, label1, label2))
                    if verbose:
                        print(f"⚠️  OVERLAP in axes[{ax_idx}]: {label1} ↔ {label2}")

    if verbose and not all_overlaps:
        print("✓ No overlaps detected")

    return all_overlaps


def check_and_save(fig, filepath, dpi=200, raise_on_overlap=False):
    """
    Check for overlaps before saving. Warns or raises if overlaps found.

    Usage:
        fig, ax = plt.subplots()
        # ... create figure ...
        check_and_save(fig, 'my_figure.png')
    """
    # Need to draw first to get accurate text positions
    fig.canvas.draw()

    overlaps = check_overlaps(fig, verbose=True)

    if overlaps and raise_on_overlap:
        raise ValueError(f"Figure has {len(overlaps)} overlapping elements. Fix before saving.")

    fig.savefig(filepath, dpi=dpi, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {filepath}")

    return overlaps


# =============================================================================
# SIGNATURE COLOUR PALETTE
# =============================================================================

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


def hex_to_rgb(hex_color):
    """Convert hex to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))


# Create custom thermal colormap: cold (0) -> hot (1)
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal_signature',
    [
        (0.0, hex_to_rgb(PALETTE['cold'])),
        (0.25, hex_to_rgb(PALETTE['cool'])),
        (0.5, hex_to_rgb(PALETTE['neutral'])),
        (0.75, hex_to_rgb(PALETTE['warm'])),
        (1.0, hex_to_rgb(PALETTE['hot']))
    ]
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
        x_wave = np.linspace(0, 0.18, 35)
        y_wave = y_pos + 0.07 * np.sin(x_wave * 38) * (1 - x_wave/0.18)
        x_wave = x_start + direction * x_wave
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

    for i in range(n_segments):
        y0 = y_bottom + i * height / n_segments
        y1 = y_bottom + (i + 1) * height / n_segments
        frac = i / n_segments
        color = THERMAL_CMAP(frac)
        ax.fill([x, x, x + width, x + width], [y0, y1, y1, y0],
                color=color, edgecolor='none')

    ax.plot([x, x], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.2)
    ax.plot([x + width, x + width], [y_bottom, y_bottom + height],
            color=PALETTE['text'], linewidth=1.2)

    for frac, temp in [(0, t_min), (0.5, (t_min+t_max)/2), (1, t_max)]:
        y = y_bottom + frac * height
        ax.plot([x + width, x + width + 0.015], [y, y],
                color=PALETTE['text'], linewidth=1)
        ax.text(x + width + 0.025, y, f'{temp:.0f}°C', fontsize=8,
                va='center', color=PALETTE['text'])


def draw_temp_label(ax, x, y, label, temp_type='hot', position='above'):
    """
    Draw a temperature label with pill background, properly spaced.

    Args:
        ax: matplotlib axes
        x, y: position of the surface/point being labelled
        label: text label (e.g., r'$T_1$')
        temp_type: 'hot' or 'cold' - determines colour
        position: 'above', 'below', 'left', 'right' - where to place label
    """
    color = PALETTE['hot'] if temp_type == 'hot' else PALETTE['cold']
    desc = 'hot surface' if temp_type == 'hot' else 'cold surface'

    # Offsets based on position
    offsets = {
        'above': (0, 0.5, 0, 0.85, 'center', 'center'),
        'below': (0, -0.5, 0, -0.85, 'center', 'center'),
        'left': (-0.3, 0, -0.5, 0, 'right', 'center'),
        'right': (0.3, 0, 0.5, 0, 'left', 'center'),
    }

    dx1, dy1, dx2, dy2, ha, va = offsets[position]

    # Main label with pill
    ax.text(x + dx1, y + dy1, label, fontsize=15, ha=ha, va=va,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                     edgecolor='none'), zorder=10)

    # Description text
    ax.text(x + dx2, y + dy2, desc, fontsize=10, ha=ha, va=va,
            color=color, style='italic')


# =============================================================================
# QUICK VALIDATION
# =============================================================================

if __name__ == '__main__':
    # Test the overlap checker
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create intentional overlap
    ax.text(0.5, 0.5, 'Label 1', fontsize=14, ha='center')
    ax.text(0.52, 0.52, 'Label 2', fontsize=14, ha='center')  # Overlaps!
    ax.text(0.1, 0.1, 'Label 3', fontsize=14)  # No overlap

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    print("Testing overlap detection...")
    fig.canvas.draw()
    overlaps = check_overlaps(fig)

    if overlaps:
        print(f"\nFound {len(overlaps)} overlap(s) - checker working correctly!")

    plt.close()
