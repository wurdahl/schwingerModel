"""Paper-ready plots from bootstrapEnsemble output.

House style (palette, fonts, thick lines, white-filled markers, hand-made
legends) is documented in PLOTTING.md; setStyle() applies it. Plot functions
apply it automatically on first use, so a bare import-and-plot already produces
a figure that can go straight into a paper.

Every function here consumes bootstrapEnsemble output verbatim — the
[central, err, cov] triple — so notebooks never unpack bootstrap internals.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

# imported rather than reimplemented: the overlaid curve must be the SAME model
# massReduce fitted, or the plot quietly misrepresents the fit
from .GEVP import _backwardFactor, _FIT_SIGNS


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


# Main JLab palette — first choice, in this order
JLab_red    = rgb_to_hex((192, 39, 45))
JLab_orange = rgb_to_hex((249, 102, 0))
JLab_blue   = rgb_to_hex((47, 122, 121))
JLab_green  = rgb_to_hex((65, 125, 10))

# Extended palette — only when the main four run out
bright_blue     = rgb_to_hex((0, 109, 219))
rose_pink       = rgb_to_hex((255, 109, 182))
lavender_violet = rgb_to_hex((182, 109, 255))
burnt_orange    = rgb_to_hex((219, 109, 0))

PALETTE = [JLab_blue, JLab_red, JLab_orange, JLab_green,
           bright_blue, rose_pink, lavender_violet, burnt_orange]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

LW = 2.5        # standard curve linewidth
GREY = "0.8"    # reference lines (zero, thresholds, asymptotes)
BAND = "0.93"   # shaded regions (fit windows)

# white-filled markers, thick everything, no connecting line.
# capsize: caps span 2*capsize points and the symbol spans ms points, so caps
# need 2*capsize >= ms + ~6pt to stay visible PAST the marker's sides even when
# the whole error bar is smaller than the symbol — otherwise tiny errors vanish
# behind the white fill and read as "no error measured". At ms=8 that means
# capsize >= 7; keep the margin if ms ever grows.
eb_kw = dict(ms=8, mfc="white", mew=2.0, elinewidth=2.0, capsize=7.0,
             capthick=2.0, ls="none", zorder=3)

# Backing for hand-placed text. Fully transparent by default: any nonzero alpha
# still reads as a box where the patch crosses the shaded fit window, because
# BAND is only a few percent off white. Place the text in clear space instead.
# Raise the alpha only for a label that has to sit on dense markers.
TEXTBOX = dict(facecolor="white", edgecolor="none", alpha=0.0, pad=2)

_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",       # Computer Modern math, matches the papers
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.spines.top": False,       # no box — only the axes you read off of
    "axes.spines.right": False,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,             # keep text as text for Keynote editing
    "svg.fonttype": "none",
}

_styleApplied = False


def setStyle(quietFonts=True, **overrides):
    """Apply the house rcParams globally (see PLOTTING.md).

    Plot functions in this module call this on first use, so an explicit call is
    only needed for hand-rolled figures in a notebook.

    Args:
        quietFonts: Silence fontTools' "'created' timestamp seems very low"
            warnings. The Computer Modern fonts matplotlib ships (cmr10, cmmi10,
            cmsy10 — the ones mathtext.fontset="cm" uses) store head-table
            timestamps that predate what fontTools expects, so each figure logs
            two lines per font. Purely cosmetic, and not worth giving up CM
            math. Pass False to hear them again. Defaults to True.
        **overrides: Extra rcParams applied on top of the house defaults.
    """
    global _styleApplied
    if quietFonts:
        logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
    plt.rcParams.update(_RC)
    if overrides:
        plt.rcParams.update(overrides)
    _styleApplied = True


def _ensureStyle():
    if not _styleApplied:
        setStyle()


def fmtErr(v, e):
    """Format as 0.7871(12) — value with parenthesized error to 2 sig figs.

    Args:
        v: Central value.
        e: Error (symmetric, or a symmetrized asymmetric error).

    Returns:
        str: LaTeX-safe string; "no fit" for a non-finite central value and a
        plain 4-decimal value when the error is non-finite or non-positive.
    """
    if not np.isfinite(v):
        return r"\mathrm{no\ fit}"
    if not np.isfinite(e) or e <= 0:
        return f"{v:.4f}"
    ndig = max(-int(np.floor(np.log10(e))) + 1, 0)
    return f"{v:.{ndig}f}({round(e * 10**ndig):d})"


# ---------------------------------------------------------------------------
# Panel grid
# ---------------------------------------------------------------------------

def makeGrid(n, maxCols=4, panelSize=(4.0, 4.2)):
    """Lay out n panels, wrapping at maxCols per row.

    Capped at 4 columns because a wider row shrinks below single-column paper
    width; extra panels wrap to further rows, which costs less vertical space
    than one panel per row.

    Args:
        n: Number of panels.
        maxCols: Maximum panels per row. Defaults to 4.
        panelSize: (width, height) in inches per panel. Defaults to (4.0, 4.2).

    Returns:
        tuple: (fig, axes, nRows, nCols) with axes a flat (n,) object array of
        the used axes; any trailing unused axes are already hidden.
    """
    _ensureStyle()
    nCols = min(n, maxCols)
    nRows = int(np.ceil(n / nCols))
    fig, axes = plt.subplots(nRows, nCols,
                             figsize=(panelSize[0] * nCols, panelSize[1] * nRows),
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.12)
    flat = axes.ravel()
    for ax in flat[n:]:                       # ragged last row
        ax.set_visible(False)
    return fig, flat[:n], nRows, nCols


def _isSeq(x):
    return hasattr(x, "__len__") and not isinstance(x, (str, bytes))


def _panelWindows(fitT, nPanels):
    """Resolve a fitT argument into one massReduce-style spec per panel.

    Accepted forms, by nesting depth:
      (lo, hi)                      one window, shared by every panel
      [(lo, hi), ...]               a flat list of windows — read as ONE PER
                                    PANEL when its length matches the panel
                                    count, otherwise as one per state, shared
      [[(lo, hi), ...], ...]        one per-state list per panel

    The flat case is genuinely ambiguous when the panel and state counts
    coincide; per-panel wins, since that is the axis these plots lay out. To
    force the per-state reading there, nest it explicitly: [perState] * nPanels.

    Args:
        fitT: One of the forms above, or None.
        nPanels: Number of panels being drawn.

    Returns:
        list: nPanels entries, each None, a (lo, hi) tuple, or a list of them.

    Raises:
        ValueError: If a nested per-panel list has the wrong length, or the
            nesting is deeper than the forms above.
    """
    if fitT is None:
        return [None] * nPanels

    depth, probe = 0, fitT
    while _isSeq(probe):
        depth += 1
        probe = probe[0]

    if depth == 1:                                     # (lo, hi)
        return [tuple(fitT)] * nPanels
    if depth == 2:                                     # list of windows
        if len(fitT) == nPanels:                       # one per panel
            return [tuple(w) for w in fitT]
        return [[tuple(w) for w in fitT]] * nPanels     # per state, shared
    if depth == 3:                                     # per-state list per panel
        if len(fitT) != nPanels:
            raise ValueError(f"fitT has {len(fitT)} per-panel entries but there "
                             f"are {nPanels} panels")
        return [[tuple(w) for w in spec] for spec in fitT]
    raise ValueError(f"fitT nested {depth} deep; expected (lo, hi), a list of "
                     "those, or a per-panel list of those")


def _window(spec, state):
    """Pick `state`'s window out of one panel's spec from _panelWindows."""
    if spec is None:
        return None
    return tuple(spec[state]) if _isSeq(spec[0]) else tuple(spec)


def fitCurve(form="exp", ti=1, shift=0, dimt=None):
    """Rebuild massReduce's fitted model as f(t, E, logA), for overlaying.

    Pass the SAME form/ti/shift you gave massReduce. With the default "exp" this
    is just exp(logA - E t); the periodic forms multiply in the around-the-torus
    image, so the drawn curve turns over where the data does instead of running
    off the bottom of the plot.

    Args:
        form: "exp", "cosh", "sinh", or "auto" ("sinh" if shift else "cosh").
        ti: GEVP reference slice, so actual time is tau = t + ti. Defaults to 1.
        shift: The shift used to build the curves. Defaults to 0.
        dimt: Full temporal extent T. Required for the periodic forms.

    Returns:
        Callable[[array, float, float], array]: f(t, E, logA), carrying a
        `.periodic` flag the plot functions use to decide how far to draw it.

    Raises:
        ValueError: On an unknown form, or a periodic form without dimt.
    """
    if form == "auto":
        form = "sinh" if shift else "cosh"
    if form not in _FIT_SIGNS:
        raise ValueError(f"form {form!r} not in {sorted(_FIT_SIGNS)} or 'auto'")
    sign = _FIT_SIGNS[form]
    if sign != 0 and dimt is None:
        raise ValueError(f"dimt is required for the {form!r} form")

    def _model(t, E, logA):
        t = np.asarray(t, dtype=float)
        return np.exp(logA - E * t) * _backwardFactor(E, t, ti, dimt, shift, sign)

    _model.periodic = sign != 0
    return _model


_EXP_MODEL = fitCurve("exp")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def prinCorrelSemilog(prinCorrels, masses=None, labels=None, states=None,
                      fitT=None, tmax=None, absolute=True, fitModel=None,
                      maxCols=4, title=None,
                      xlabel=r"$(t - t_i)/a$",
                      ylabel=None, ylim=None, savePath=None):
    """Raw GEVP eigenvalue curves on a log axis, one panel per ensemble.

    The diagnostic view: a clean principal correlator is a straight line here,
    so curvature (excited-state contamination early, thermal pollution late) and
    the point where errors swallow the signal are both read off directly. Use
    prinCorrelOverFit once a fit exists and the question is whether it describes
    the data.

    By default the magnitude is plotted, so the whole time extent stays visible
    on a log axis. A curve that dives toward zero and climbs back has changed
    sign there, which is the cusp of a sinh: with shift > 0 the correlator
    C(t + shift) - C(t) is antisymmetric about T/2 and must cross zero there.
    Points below zero are drawn with filled markers, against the house
    white-filled default, so the sign is readable and not merely inferred.

    Args:
        prinCorrels: List of bootstrapEnsemble outputs run with a
            makeGevpReduce reduce: each [central, err, cov] with central
            (T', n_states) and err (2, T', n_states) rows
            (high - central, central - low).
        masses: Optional list of bootstrapEnsemble outputs from
            massReduce(..., withAmp=True), aligned with prinCorrels. If given,
            each state's fitted exponential exp(logA - E t) is drawn through its
            points, which is what makes a bad fit window obvious. The line spans
            only the pre-crossing region, where a forward exponential applies.
            Defaults to None (data only).
        labels: Per-panel titles, e.g. [r"$m_0 = 0.1$", ...]. Defaults to None.
        states: Which GEVP states to draw, e.g. [0, 1]. Defaults to None (all).
        fitT: The window(s) given to massReduce. Either one (lo, hi) shared by
            every panel, a flat list of windows (one per panel when its length
            matches the panel count, else one per state), or a per-panel list of
            per-state lists — see _panelWindows. Shades [lo, hi - 1] using the
            first drawn state's window. Defaults to None (no shading).
        tmax: Plot only t < tmax. Defaults to None, which uses the whole curve
            when absolute is True, and otherwise stops shortly after the last
            positive point since a log axis cannot draw the rest.
        absolute: Plot |lambda| rather than lambda. Defaults to True. Set False
            to drop negative points entirely and see only the decaying side.
        fitModel: f(t, E, logA) rebuilding the model massReduce fitted; build it
            with fitCurve(form, ti, shift, dimt) using the SAME arguments. The
            default assumes a plain forward exponential, so pass this whenever
            massReduce ran with fitForm="cosh"/"sinh" — otherwise the overlaid
            line is not the curve that was fitted.
        maxCols: Maximum panels per row. Defaults to 4.
        title: Figure suptitle. Defaults to None.
        xlabel: Time-axis label, on the bottom panel of each column.
        ylabel: Left-column label. Defaults to None, which follows `absolute`.
        ylim: (lo, hi) shared by every panel. Defaults to None (per-panel
            autoscale); pass one to compare decay rates across ensembles.
        savePath: If given, savefig here (use a .pdf under figs/).

    Returns:
        tuple: (fig, axes) with axes the flat (n,) array of used axes.

    Raises:
        ValueError: If masses is given and differs in length from prinCorrels.
    """
    if masses is not None:
        if len(masses) != len(prinCorrels):
            raise ValueError(f"prinCorrels has {len(prinCorrels)} entries but masses "
                             f"has {len(masses)}; they must be aligned")
        for i, m in enumerate(masses):
            if np.ndim(m[0]) != 2 or np.shape(m[0])[1] != 2:
                raise ValueError(f"masses[{i}] central has shape {np.shape(m[0])}; "
                                 "expected (n_states, 2) — rerun with "
                                 "massReduce(..., withAmp=True)")

    n = len(prinCorrels)
    fig, axes, nRows, nCols = makeGrid(n, maxCols=maxCols)
    specs = _panelWindows(fitT, n)

    for i, ax in enumerate(axes):
        central, err = prinCorrels[i][0], prinCorrels[i][1]
        sts = range(central.shape[1]) if states is None else states

        t = np.arange(central.shape[0])

        if tmax is not None:
            tHi = tmax
        elif absolute:
            tHi = central.shape[0]            # |.| keeps every slice drawable
        else:
            # Stop where the last drawn state goes non-positive: a log axis
            # cannot show those points, and with shift > 0 the sign flips at
            # T/2, so the tail is structurally empty rather than noisy.
            posAny = np.zeros(central.shape[0], dtype=bool)
            for s in sts:
                posAny |= central[:, s] > 0
            lastPos = np.max(np.nonzero(posAny)) if posAny.any() else central.shape[0] - 1
            tHi = min(central.shape[0], lastPos + 3)
        sel = t < tHi

        win = _window(specs[i], list(sts)[0])
        if win is not None:
            ax.axvspan(win[0], win[1] - 1, color=BAND, zorder=0)

        dataLo, dataHi = np.inf, -np.inf
        for k, s in enumerate(sts):
            color = PALETTE[k % len(PALETTE)]
            marker = MARKERS[k % len(MARKERS)]

            raw = central[sel, s]
            eHi, eLo = np.clip(err[:, sel, s], 0.0, None)   # [high-central, central-low]
            neg = raw < 0
            if absolute:
                y = np.abs(raw)
                # |.| reflects the interval about zero, so for a negative point
                # its upper and lower bars swap roles
                errUp = np.where(neg, eLo, eHi)
                errDn = np.where(neg, eHi, eLo)
            else:
                y, errUp, errDn = raw, eHi, eLo
            # a lower bar reaching <= 0 has no home on a log axis; keep it just
            # inside the point so the bar still shows its full upper extent
            errDn = np.minimum(errDn, y * (1 - 1e-9))
            good = y > 0

            # positive and negative drawn separately: filled markers flag the
            # sign that taking |.| would otherwise hide
            for mask, face in ((good & ~neg, "white"), (good & neg, color)):
                if mask.any():
                    kw = dict(eb_kw, mfc=face)
                    ax.errorbar(t[sel][mask], y[mask],
                                yerr=[errDn[mask], errUp[mask]],
                                marker=marker, color=color, mec=color, **kw)
            if good.any():
                dataLo = min(dataLo, np.nanmin(y[good]))
                dataHi = max(dataHi, np.nanmax((y + errUp)[good]))

            if masses is not None:
                E, logA = masses[i][0][s]
                if np.isfinite(E) and np.isfinite(logA):
                    model = fitModel if fitModel is not None else _EXP_MODEL
                    if getattr(model, "periodic", False):
                        tf = t[sel]         # turns over on its own; draw it all
                    else:
                        # a bare forward exponential only applies pre-crossing;
                        # continuing it would drag the log range down by decades
                        fitSel = (raw > 0) & ~np.maximum.accumulate(neg)
                        tf = t[sel][fitSel] if fitSel.any() else t[sel]
                    ax.plot(tf, model(tf, E, logA), color=color, lw=LW, zorder=2)

        ax.set_yscale("log")
        # scale to the DATA; late-time bars span decades and would set the range
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif np.isfinite(dataLo) and np.isfinite(dataHi) and dataHi > dataLo:
            ax.set_ylim(dataLo * 0.2, dataHi * 5)
        if labels is not None:
            ax.set_title(labels[i], fontsize=18, pad=10)
        ax.set_xlim(-0.6, tHi - 0.4)

        if i + nCols >= n:                  # nothing below it — label the x axis
            ax.set_xlabel(xlabel)
        if i % nCols == 0:                  # leftmost of its row
            ax.set_ylabel(ylabel if ylabel is not None else
                          (r"$|\lambda^{(s)}(t)|$" if absolute
                           else r"$\lambda^{(s)}(t)$"))

    # one hand-made legend, in the first panel only
    stateList = list(range(prinCorrels[0][0].shape[1]) if states is None else states)
    for k, s in enumerate(stateList):
        axes[0].text(0.97, 0.95 - 0.085 * k, rf"$\lambda^{{({s})}}$",
                     color=PALETTE[k % len(PALETTE)], ha="right", va="top",
                     transform=axes[0].transAxes, fontsize=15, zorder=5,
                     bbox=TEXTBOX)

    if title is not None:
        fig.suptitle(title, fontsize=18)
    if savePath is not None:
        fig.savefig(savePath)
    return fig, axes


def prinCorrelOverFit(prinCorrels, masses, labels=None, state=0, fitT=None,
                      tmax=None, fitModel=None, maxCols=4, color=JLab_blue,
                      marker="o", title=None, xlabel=r"$(t - t_i)/a$",
                      ylabel=None, savePath=None):
    """Principal correlators divided by their fitted model, one panel each.

    Plots lambda^(s)(t) / fit(t), which flattens to 1 wherever the fit
    describes the data — the deviation from unity is far easier to read than a
    log-scale correlator, and the panels stay comparable across ensembles even
    when their masses differ by an order of magnitude.

    Args:
        prinCorrels: List of bootstrapEnsemble outputs run with a
            makeGevpReduce reduce: each [central, err, cov] with central
            (T', n_states) and err (2, T', n_states) rows
            (high - central, central - low).
        masses: List of bootstrapEnsemble outputs run with
            massReduce(..., withAmp=True), aligned with prinCorrels: each
            central is (n_states, 2) with columns [E, logA]. withAmp=True is
            required — the amplitude is what sets the curve being divided out.
        labels: Per-panel titles, e.g. [r"$N_x = 4$", ...]. Defaults to None
            (no panel titles).
        state: Which GEVP state to plot, 0 = ground. Defaults to 0.
        fitT: The window(s) given to massReduce (half-open, in curve-index
            units). Either one (lo, hi) shared by every panel, a flat list of
            windows (one per panel when its length matches the panel count, else
            one per state), or a per-panel list of per-state lists — see
            _panelWindows. Shades [lo, hi - 1] and focuses the y-range there.
            Defaults to None (no shading, y-range from all plotted points).
        tmax: Plot only t < tmax, to cut the late-time noise tail. Defaults to
            None (whole curve).
        fitModel: f(t, E, logA) rebuilding the model massReduce fitted; build it
            with fitCurve(form, ti, shift, dimt) using the SAME arguments. The
            default divides by a plain forward exponential, so pass this
            whenever massReduce ran with fitForm="cosh"/"sinh" — otherwise the
            ratio bakes the periodic image into the deviation from 1.
        maxCols: Maximum panels per row. Defaults to 4.
        color: Marker/errorbar color. Defaults to JLab_blue.
        marker: Marker shape. Defaults to "o".
        title: Figure suptitle, e.g. the ensemble parameters. Working figures
            want one; strip it for the paper (that goes in the caption).
            Defaults to None.
        xlabel: Time-axis label, drawn on the bottom panel of each column.
            Defaults to r"$(t - t_i)/a$".
        ylabel: Left-column label. Defaults to None, which builds
            lambda^(state)(t) / A e^{-E_state t}.
        savePath: If given, savefig here (use a .pdf under figs/). Defaults to
            None.

    Returns:
        tuple: (fig, axes) with axes the flat (n,) array of used axes.

    Raises:
        ValueError: If the two lists differ in length, or masses was not built
            with withAmp=True.
    """
    if len(prinCorrels) != len(masses):
        raise ValueError(f"prinCorrels has {len(prinCorrels)} entries but masses "
                         f"has {len(masses)}; they must be aligned")

    n = len(prinCorrels)
    fig, axes, nRows, nCols = makeGrid(n, maxCols=maxCols)
    specs = _panelWindows(fitT, n)

    for i, ax in enumerate(axes):
        win = _window(specs[i], state)
        central, err = prinCorrels[i][0], prinCorrels[i][1]
        mCentral, mErr = masses[i][0], masses[i][1]

        if np.ndim(mCentral) != 2 or np.shape(mCentral)[1] != 2:
            raise ValueError(f"masses[{i}] central has shape {np.shape(mCentral)}; "
                             "expected (n_states, 2) — rerun with "
                             "massReduce(..., withAmp=True)")

        E, logA = mCentral[state]
        dE = np.nanmax(mErr[:, state, 0])            # widest of the two error rows

        t = np.arange(central.shape[0])              # curve-index time, matches the fit
        sel = t < tmax if tmax is not None else np.ones_like(t, dtype=bool)

        ax.axhline(1.0, color=GREY, lw=LW, zorder=0)
        if win is not None:
            ax.axvspan(win[0], win[1] - 1, color=BAND, zorder=0)   # fit window

        if np.isfinite(E) and np.isfinite(logA):
            model = fitModel if fitModel is not None else _EXP_MODEL
            fit = model(t, E, logA)
            # a sinh model passes through zero at its crossing; the ratio is
            # meaningless within noise of that point, so blank it rather than
            # letting one near-zero division set the panel's scale
            fit[np.abs(fit) < 1e-12] = np.nan
            # the periodic factor inside fitCurve carries |.| (the mass fit runs
            # in log space), so ratio |data| against it: past a sinh crossing the
            # ratio then returns to +1 instead of sitting at -1 off-panel
            ratio = (np.abs(central[:, state]) / fit)[sel]
            # err rows are [high-central, central-low]; matplotlib wants
            # [lower, upper]. A negative entry means the central value fell
            # outside its own bootstrap band (bias at noisy t) — clip that side
            # to zero so the marker still shows the central value.
            errUp, errDn = np.clip(err[:, sel, state], 0.0, None) / fit[sel]
            ax.errorbar(t[sel], ratio, yerr=[errDn, errUp],
                        marker=marker, color=color, mec=color, **eb_kw)

            # y-range from the band around the fit region — late-time noise
            # would otherwise dominate the autoscale — with headroom for the
            # aE label
            focus = (t[sel] <= win[1]) if win is not None else np.ones(sel.sum(), dtype=bool)
            if focus.any():
                lo = np.nanmin((ratio - errDn)[focus])
                hi = np.nanmax((ratio + errUp)[focus])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    pad = 0.10 * (hi - lo)
                    ax.set_ylim(lo - pad, hi + 3.0 * pad)

        ax.text(0.95, 0.95, rf"$aE_{{{state}}} = {fmtErr(E, dE)}$",
                ha="right", va="top", transform=ax.transAxes, fontsize=15,
                zorder=5, bbox=TEXTBOX)

        if labels is not None:
            ax.set_title(labels[i], fontsize=18, pad=10)
        tHi = (tmax if tmax is not None else central.shape[0])
        ax.set_xlim(-0.6, tHi - 0.4)
        ax.set_xticks(range(0, tHi, 2 if tHi <= 20 else 5))

        if i + nCols >= n:                  # nothing below it — label the x axis
            ax.set_xlabel(xlabel)
        if i % nCols == 0:                  # leftmost of its row
            # the exponential-specific label would misdescribe a periodic fit
            defaultY = (rf"$\lambda^{{({state})}}(t) \, / \, \mathrm{{fit}}$"
                        if fitModel is not None and getattr(fitModel, "periodic", False)
                        else rf"$\lambda^{{({state})}}(t) \, / \, A\,e^{{-E_{state} t}}$")
            ax.set_ylabel(ylabel if ylabel is not None else defaultY)

    if title is not None:
        fig.suptitle(title, fontsize=18)
    if savePath is not None:
        fig.savefig(savePath)
    return fig, axes
