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

    return all_overlaps


def check_margins(fig, margin_frac=0.05, verbose=True):
    """
    Check that no text element sits too close to the axis boundary.

    Flags any text whose bounding box (in data coords) is within
    *margin_frac* of the axis limits on any side.  This catches labels
    placed on or near drawing elements that line up with the boundary
    (e.g. vessel walls, rectangles drawn at the axis edge).

    Parameters
    ----------
    fig : matplotlib Figure
    margin_frac : float
        Fraction of the axis span that defines the "danger zone" at each
        edge.  Default 0.05 (5 %).
    verbose : bool
        If True, print each issue found.

    Returns
    -------
    list of str
        One entry per offending element, e.g.
        ``"Text '$T_\\infty$' within 5% of left edge in axes[0]"``
    """
    renderer = fig.canvas.get_renderer()
    issues = []

    for ax_idx, ax in enumerate(fig.axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_margin = (xlim[1] - xlim[0]) * margin_frac
        y_margin = (ylim[1] - ylim[0]) * margin_frac

        # Collect data-annotation text only (skip axis labels, title, tick labels)
        # Axis labels and titles are expected to be outside the data frame;
        # including them causes false positives that force strict=False,
        # which then silences real issues.
        axis_labels = {id(ax.xaxis.label), id(ax.yaxis.label), id(ax.title)}
        texts = []
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible():
                texts.append(t)

        for t in texts:
            try:
                bp = t.get_bbox_patch()
                if bp is not None:
                    bbox = bp.get_window_extent(renderer).transformed(
                        ax.transData.inverted())
                else:
                    bbox = t.get_window_extent(renderer).transformed(
                        ax.transData.inverted())

                name = t.get_text()[:25]
                edges_hit = []
                if bbox.x0 < xlim[0] + x_margin:
                    edges_hit.append('left')
                if bbox.x1 > xlim[1] - x_margin:
                    edges_hit.append('right')
                if bbox.y0 < ylim[0] + y_margin:
                    edges_hit.append('bottom')
                if bbox.y1 > ylim[1] - y_margin:
                    edges_hit.append('top')

                if edges_hit:
                    msg = (f"Text '{name}' within {margin_frac:.0%} of "
                           f"{'/'.join(edges_hit)} edge in axes[{ax_idx}]")
                    issues.append(msg)
                    if verbose:
                        print(f"⚠️  MARGIN: {msg}")
            except Exception:
                pass

    if verbose and not issues:
        print("✓ No margin issues detected")

    return issues


def check_frame_bounds(fig, verbose=True):
    """
    Check that no text element or annotation is positioned outside the axis
    limits (xlim/ylim).  This is the primary defence against labels that end
    up in the whitespace area outside the figure frame.

    For figures with ax.axis('off'), there is no visible border, so this
    code-based check is the only reliable way to detect boundary violations.
    Elements that are outside xlim/ylim but still appear in the saved image
    (due to bbox_inches='tight') are still defects — they are in the wrong
    place.

    Parameters
    ----------
    fig : matplotlib Figure
    verbose : bool
        If True, print each issue found.

    Returns
    -------
    list of str
        One entry per offending element.
    """
    renderer = fig.canvas.get_renderer()
    issues = []

    for ax_idx, ax in enumerate(fig.axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        margin = 0.05  # 5% proximity threshold

        # Collect data-annotation text only (skip axis labels and title —
        # they are expected to sit outside the data frame by design)
        texts = []
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible():
                texts.append(t)

        for t in texts:
            try:
                bp = t.get_bbox_patch()
                if bp is not None:
                    bbox = bp.get_window_extent(renderer).transformed(
                        ax.transData.inverted())
                else:
                    bbox = t.get_window_extent(renderer).transformed(
                        ax.transData.inverted())

                name = t.get_text()[:25]

                # Check if element is OUTSIDE the frame
                outside = []
                if bbox.x1 < xlim[0]:
                    outside.append(f'fully left of xlim ({bbox.x1:.2f} < {xlim[0]:.2f})')
                elif bbox.x0 < xlim[0]:
                    outside.append(f'partially left of xlim ({bbox.x0:.2f} < {xlim[0]:.2f})')
                if bbox.x0 > xlim[1]:
                    outside.append(f'fully right of xlim ({bbox.x0:.2f} > {xlim[1]:.2f})')
                elif bbox.x1 > xlim[1]:
                    outside.append(f'partially right of xlim ({bbox.x1:.2f} > {xlim[1]:.2f})')
                if bbox.y1 < ylim[0]:
                    outside.append(f'fully below ylim ({bbox.y1:.2f} < {ylim[0]:.2f})')
                elif bbox.y0 < ylim[0]:
                    outside.append(f'partially below ylim ({bbox.y0:.2f} < {ylim[0]:.2f})')
                if bbox.y0 > ylim[1]:
                    outside.append(f'fully above ylim ({bbox.y0:.2f} > {ylim[1]:.2f})')
                elif bbox.y1 > ylim[1]:
                    outside.append(f'partially above ylim ({bbox.y1:.2f} > {ylim[1]:.2f})')

                if outside:
                    msg = (f"Text '{name}' is OUTSIDE FRAME: "
                           f"{'; '.join(outside)} in axes[{ax_idx}]")
                    issues.append(msg)
                    if verbose:
                        print(f"🚨 FRAME VIOLATION: {msg}")
                    continue  # Don't also check proximity if already outside

                # Check if element is JAMMED against an edge (within 5%)
                jammed = []
                if bbox.x0 < xlim[0] + margin * x_range:
                    jammed.append('left')
                if bbox.x1 > xlim[1] - margin * x_range:
                    jammed.append('right')
                if bbox.y0 < ylim[0] + margin * y_range:
                    jammed.append('bottom')
                if bbox.y1 > ylim[1] - margin * y_range:
                    jammed.append('top')

                if jammed:
                    msg = (f"Text '{name}' jammed against "
                           f"{'/'.join(jammed)} edge in axes[{ax_idx}]")
                    issues.append(msg)
                    if verbose:
                        print(f"⚠️  BOUNDARY: {msg}")
            except Exception:
                pass

        # Also check FancyArrowPatch logical endpoints (posA, posB).
        # NOTE: get_path() returns internal bezier vertices including control
        # points that can extend far beyond the visible arrow — always use
        # _posA_posB instead, which gives the actual data-coordinate endpoints.
        for child in ax.get_children():
            if isinstance(child, FancyArrowPatch):
                try:
                    posAB = getattr(child, '_posA_posB', None)
                    if posAB is None:
                        continue
                    for label, (px, py) in [('start', posAB[0]),
                                            ('end', posAB[1])]:
                        outside = []
                        if px < xlim[0]:
                            outside.append(f'left of xlim ({px:.2f} < {xlim[0]:.2f})')
                        if px > xlim[1]:
                            outside.append(f'right of xlim ({px:.2f} > {xlim[1]:.2f})')
                        if py < ylim[0]:
                            outside.append(f'below ylim ({py:.2f} < {ylim[0]:.2f})')
                        if py > ylim[1]:
                            outside.append(f'above ylim ({py:.2f} > {ylim[1]:.2f})')
                        if outside:
                            msg = (f"Arrow {label} at ({px:.2f}, {py:.2f}) "
                                   f"is outside frame: {'; '.join(outside)} "
                                   f"in axes[{ax_idx}]")
                            issues.append(msg)
                            if verbose:
                                print(f"🚨 FRAME VIOLATION: {msg}")
                except Exception:
                    pass

    if verbose and not issues:
        print("✓ All elements within frame bounds")

    return issues


def check_structural_bounds(fig, margin_frac=0.02, verbose=True):
    """
    For ax.axis('off') figures, check that all labels and annotations sit
    inside the visual boundary defined by the structural drawing elements.

    The problem this solves: with ax.axis('off'), the reader perceives the
    major drawing elements (vessel walls, slabs, containers) as the figure
    boundary — not the invisible axis limits.  An element can be inside the
    axis frame but visually outside the diagram.

    Structural elements are identified by type and appearance:
      - Rectangle, Polygon, PathPatch, RegularPolygon  (always structural)
      - Circle / Ellipse with alpha >= 0.3             (solid shapes)
      - Line2D                                          (edges, boundaries)
      - FancyArrowPatch endpoints                       (arrows define diagram reach)
    Non-structural (excluded):
      - Circle / Ellipse with alpha < 0.3              (glow effects)
      - FancyBboxPatch                                  (text pill backgrounds, equation boxes)

    Parameters
    ----------
    fig : matplotlib Figure
    margin_frac : float
        Padding around the structural bounding box, as a fraction of its
        span.  Default 0.02 (2 %).  Labels are allowed within this slim
        margin beyond the structural elements; anything further out is
        visually outside the diagram.
    verbose : bool

    Returns
    -------
    list of str
        One entry per label outside the structural boundary.
    """
    from matplotlib.patches import (Rectangle, Polygon, PathPatch,
                                    RegularPolygon, Ellipse)
    from matplotlib.lines import Line2D

    renderer = fig.canvas.get_renderer()
    issues = []

    for ax_idx, ax in enumerate(fig.axes):
        # Only applies to axis('off') figures where the frame is invisible
        if ax.axison:
            continue

        # --- Step 1: Find structural elements and compute visual boundary ---
        struct_xs, struct_ys = [], []

        for p in ax.patches:
            if not p.get_visible():
                continue

            # Classify the patch
            is_structural = False

            if isinstance(p, (Rectangle, Polygon, PathPatch, RegularPolygon)):
                is_structural = True
            elif isinstance(p, (Circle, Ellipse)):
                # Solid shapes are structural; faint glows are not
                is_structural = p.get_alpha() is None or p.get_alpha() >= 0.3
            elif isinstance(p, FancyBboxPatch):
                # Text backgrounds and equation boxes — not structural
                is_structural = False
            elif isinstance(p, FancyArrowPatch):
                # Arrows are content but don't define the boundary
                is_structural = False

            if not is_structural:
                continue

            try:
                bbox = p.get_window_extent(renderer).transformed(
                    ax.transData.inverted())
                if bbox.width > 0 or bbox.height > 0:
                    struct_xs.extend([bbox.x0, bbox.x1])
                    struct_ys.extend([bbox.y0, bbox.y1])
            except Exception:
                pass

        # Lines (boundaries, edges, dashed control volumes)
        for line in ax.lines:
            if not line.get_visible() or not isinstance(line, Line2D):
                continue
            try:
                xdata, ydata = line.get_data()
                xdata = np.asarray(xdata, dtype=float)
                ydata = np.asarray(ydata, dtype=float)
                if len(xdata) > 0:
                    struct_xs.extend([float(np.nanmin(xdata)),
                                     float(np.nanmax(xdata))])
                    struct_ys.extend([float(np.nanmin(ydata)),
                                     float(np.nanmax(ydata))])
            except Exception:
                pass

        # FancyArrowPatch endpoints (arrows define diagram reach for slabs etc.)
        for child in ax.get_children():
            if isinstance(child, FancyArrowPatch) and child.get_visible():
                try:
                    posAB = getattr(child, '_posA_posB', None)
                    if posAB is not None:
                        for px, py in posAB:
                            struct_xs.append(px)
                            struct_ys.append(py)
                except Exception:
                    pass

        if not struct_xs or not struct_ys:
            continue

        sb_xmin, sb_xmax = min(struct_xs), max(struct_xs)
        sb_ymin, sb_ymax = min(struct_ys), max(struct_ys)
        sb_x_range = sb_xmax - sb_xmin
        sb_y_range = sb_ymax - sb_ymin

        if sb_x_range == 0 or sb_y_range == 0:
            continue

        # Add padding
        pad_x = sb_x_range * margin_frac
        pad_y = sb_y_range * margin_frac
        bound_xmin = sb_xmin - pad_x
        bound_xmax = sb_xmax + pad_x
        bound_ymin = sb_ymin - pad_y
        bound_ymax = sb_ymax + pad_y

        # --- Step 2: Check all text/annotations against structural bounds ---
        for t in ax.texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            try:
                bp = t.get_bbox_patch()
                if bp is not None:
                    bbox = bp.get_window_extent(renderer).transformed(
                        ax.transData.inverted())
                else:
                    bbox = t.get_window_extent(renderer).transformed(
                        ax.transData.inverted())

                name = t.get_text()[:25]
                outside = []
                if bbox.x0 < bound_xmin:
                    outside.append(f'left (x={bbox.x0:.2f} < {bound_xmin:.2f})')
                if bbox.x1 > bound_xmax:
                    outside.append(f'right (x={bbox.x1:.2f} > {bound_xmax:.2f})')
                if bbox.y0 < bound_ymin:
                    outside.append(f'below (y={bbox.y0:.2f} < {bound_ymin:.2f})')
                if bbox.y1 > bound_ymax:
                    outside.append(f'above (y={bbox.y1:.2f} > {bound_ymax:.2f})')

                if outside:
                    msg = (f"Text '{name}' extends beyond structural boundary: "
                           f"{', '.join(outside)} in axes[{ax_idx}]. "
                           f"Structural bounds (from drawing elements): "
                           f"x=[{sb_xmin:.1f}, {sb_xmax:.1f}] "
                           f"y=[{sb_ymin:.1f}, {sb_ymax:.1f}]")
                    issues.append(msg)
                    if verbose:
                        print(f"🚨 STRUCTURAL BOUNDS: {msg}")
            except Exception:
                pass

    if verbose and not issues:
        print("✓ All labels within structural boundary")

    return issues


def _describe_line(line, xdata, ydata, is_href, is_vref):
    """Build a human-readable description of a Line2D for overlap messages."""
    style = line.get_linestyle()
    style_desc = "dashed " if style in ('--', 'dashed') else ""
    if is_href:
        y_val = float(np.median(ydata))
        return f"{style_desc}hline at y={y_val:.1f}"
    if is_vref:
        x_val = float(np.median(xdata))
        return f"{style_desc}vline at x={x_val:.1f}"
    label = line.get_label()
    if label and not label.startswith('_'):
        return f"line '{label}'"
    return "data curve"


def check_visual_overlaps(fig, padding_frac=0.03, verbose=True):
    """
    Check for overlaps between text/patches and Line2D elements.

    Detects visual overlaps that check_overlaps (text-vs-text only) misses:
      - Labels sitting on data curves or reference lines
      - Text with bbox patches (equation boxes) intersected by curves
      - Standalone FancyBboxPatch objects intersected by curves
      - Text placed too close to horizontal/vertical reference lines

    Line2D objects with alpha < 0.3 (faint decorative lines) or
    zorder < 1 (grid lines) are excluded to avoid false positives.

    For lines with few data points (e.g. axhline with only 2 vertices),
    the path is resampled to 200 points via linear interpolation so that
    point-in-bbox checks have adequate spatial coverage.

    Parameters
    ----------
    fig : matplotlib Figure
    padding_frac : float
        Fraction of axis span to add as proximity padding around bounding
        boxes.  Points inside the unpadded bbox are reported as OVERLAP;
        points inside the padded bbox (but outside the unpadded one) are
        reported as PROXIMITY.  Default 0.03 (3 %).
    verbose : bool
        If True, print each issue found.

    Returns
    -------
    list of str
        One entry per visual overlap detected.
    """
    from matplotlib.lines import Line2D

    renderer = fig.canvas.get_renderer()
    issues = []

    for ax_idx, ax in enumerate(fig.axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if x_range == 0 or y_range == 0:
            continue

        pad_x = x_range * padding_frac
        pad_y = y_range * padding_frac

        # ---- Collect visible Line2D data in data coordinates ----
        line_data_list = []
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            alpha = line.get_alpha()
            if alpha is not None and alpha < 0.3:
                continue
            if line.get_zorder() < 1:
                continue

            try:
                path = line.get_path()
                if len(path.vertices) < 2:
                    continue
                # Transform via display coords to handle blended transforms
                # (axhline/axvline use blended axes + data transforms)
                disp = line.get_transform().transform(path.vertices)
                data = ax.transData.inverted().transform(disp)
                xd = data[:, 0].astype(float)
                yd = data[:, 1].astype(float)
                valid = np.isfinite(xd) & np.isfinite(yd)
                xd, yd = xd[valid], yd[valid]
                if len(xd) < 2:
                    continue

                # Resample sparse lines for adequate point coverage
                if len(xd) < 200:
                    t_orig = np.linspace(0, 1, len(xd))
                    t_new = np.linspace(0, 1, 200)
                    xd = np.interp(t_new, t_orig, xd)
                    yd = np.interp(t_new, t_orig, yd)

                is_href = (np.ptp(yd) < y_range * 0.001 and
                           np.ptp(xd) > x_range * 0.5)
                is_vref = (np.ptp(xd) < x_range * 0.001 and
                           np.ptp(yd) > y_range * 0.5)

                line_data_list.append((line, xd, yd, is_href, is_vref))
            except Exception:
                continue

        if not line_data_list:
            continue

        # ---- Check 1 & 3: Text (with bbox) vs Line2D ----
        texts = [t for t in ax.texts
                 if t.get_text().strip() and t.get_visible()]

        for t in texts:
            try:
                bp = t.get_bbox_patch()
                if bp is not None:
                    bbox = bp.get_window_extent(renderer).transformed(
                        ax.transData.inverted())
                else:
                    bbox = t.get_window_extent(renderer).transformed(
                        ax.transData.inverted())
                name = t.get_text()[:30]
                has_box = bp is not None

                for line, xd, yd, is_href, is_vref in line_data_list:
                    # Reference lines (axhline/axvline) span the full axis,
                    # so text near them is visually confusing at greater
                    # distances than for regular curves.  Use 2x padding.
                    is_ref = is_href or is_vref
                    eff_pad_x = pad_x * 2 if is_ref else pad_x
                    eff_pad_y = pad_y * 2 if is_ref else pad_y

                    strict = ((xd >= bbox.x0) & (xd <= bbox.x1) &
                              (yd >= bbox.y0) & (yd <= bbox.y1))
                    padded = ((xd >= bbox.x0 - eff_pad_x) &
                              (xd <= bbox.x1 + eff_pad_x) &
                              (yd >= bbox.y0 - eff_pad_y) &
                              (yd <= bbox.y1 + eff_pad_y))

                    if not np.any(padded):
                        continue

                    n_strict = int(np.sum(strict))
                    n_padded = int(np.sum(padded))

                    if n_strict > 0:
                        severity = "OVERLAP"
                        detail = f"{n_strict} line point(s) inside bbox"
                    else:
                        severity = "PROXIMITY"
                        detail = (f"{n_padded} line point(s) within "
                                  f"{padding_frac:.0%} padding")

                    line_desc = _describe_line(line, xd, yd, is_href, is_vref)
                    elem = (f"Text+box '{name}'" if has_box
                            else f"Text '{name}'")

                    msg = (f"{severity}: {elem} vs {line_desc} "
                           f"({detail}) in axes[{ax_idx}]")
                    issues.append(msg)
                    if verbose:
                        icon = "\U0001f6a8" if severity == "OVERLAP" else "\u26a0\ufe0f "
                        print(f"{icon} VISUAL {msg}")
            except Exception:
                continue

        # ---- Check 2: Standalone FancyBboxPatch vs Line2D ----
        # Text bbox patches (from ax.text(..., bbox=dict(...))) are NOT in
        # ax.patches — they are caught above via text.get_bbox_patch().
        # Only standalone patches added via ax.add_patch() appear here.
        for patch in ax.patches:
            if not isinstance(patch, FancyBboxPatch):
                continue
            if not patch.get_visible():
                continue
            try:
                bbox = patch.get_window_extent(renderer).transformed(
                    ax.transData.inverted())
                px, py = patch.get_x(), patch.get_y()
                pw, ph = patch.get_width(), patch.get_height()
                patch_name = (f"FancyBboxPatch at ({px:.1f},{py:.1f}) "
                              f"size {pw:.1f}x{ph:.1f}")

                for line, xd, yd, is_href, is_vref in line_data_list:
                    strict = ((xd >= bbox.x0) & (xd <= bbox.x1) &
                              (yd >= bbox.y0) & (yd <= bbox.y1))
                    padded = ((xd >= bbox.x0 - pad_x) & (xd <= bbox.x1 + pad_x) &
                              (yd >= bbox.y0 - pad_y) & (yd <= bbox.y1 + pad_y))

                    if not np.any(padded):
                        continue

                    n_strict = int(np.sum(strict))
                    n_padded = int(np.sum(padded))

                    if n_strict > 0:
                        severity = "OVERLAP"
                        detail = f"{n_strict} line point(s) inside patch"
                    else:
                        severity = "PROXIMITY"
                        detail = (f"{n_padded} line point(s) within "
                                  f"{padding_frac:.0%} padding")

                    line_desc = _describe_line(line, xd, yd, is_href, is_vref)
                    msg = (f"{severity}: {patch_name} vs {line_desc} "
                           f"({detail}) in axes[{ax_idx}]")
                    issues.append(msg)
                    if verbose:
                        icon = "\U0001f6a8" if severity == "OVERLAP" else "\u26a0\ufe0f "
                        print(f"{icon} VISUAL {msg}")
            except Exception:
                continue

    if verbose and not issues:
        print("\u2713 No visual element overlaps detected")

    return issues


def validate_figure(fig, strict=True):
    """
    Run all figure validation checks: overlaps, margins, and frame bounds.

    Call this once at the end of every figure code block instead of calling
    the three checkers individually.  Produces no printed output (safe for
    Quarto rendering).

    Parameters
    ----------
    fig : matplotlib Figure
    strict : bool
        If True (default), raise ValueError when any issue is found.
        This prevents Quarto from rendering a broken figure — the build
        fails and the author must fix the issue before proceeding.
        Set to False to return issues silently (e.g. for batch audits).

    Returns
    -------
    list of str
        All issues found.  Empty list means the figure passed all checks.

    Raises
    ------
    ValueError
        If strict=True and any issues are found.
    """
    fig.canvas.draw()
    issues = []

    # Overlaps (returns list of tuples — normalise to strings)
    for ax_idx, label1, label2 in check_overlaps(fig, verbose=False):
        issues.append(f"OVERLAP in axes[{ax_idx}]: {label1} ↔ {label2}")

    # Margins (returns list of strings)
    issues.extend(check_margins(fig, verbose=False))

    # Frame bounds (returns list of strings)
    issues.extend(check_frame_bounds(fig, verbose=False))

    # Visual overlaps — text/patches vs Line2D (returns list of strings)
    issues.extend(check_visual_overlaps(fig, verbose=False))

    # NOTE: check_structural_bounds was removed — it infers a visual boundary
    # from drawing elements in ax.axis('off') figures, but labels are
    # intentionally placed outside those elements (they annotate from outside).
    # Real edge/clipping issues are caught by check_margins, check_frame_bounds,
    # and Ebert's visual review of per-figure context screenshots.

    # Deduplicate (margin and frame-bound checkers can flag the same element)
    issues = list(dict.fromkeys(issues))

    if strict and issues:
        msg = f"validate_figure: {len(issues)} issue(s) found:\n"
        msg += "\n".join(f"  - {i}" for i in issues)
        raise ValueError(msg)

    return issues


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
