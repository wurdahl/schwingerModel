# Plotting style guide

House style for every figure in this repo, synthesized from the PI's
recommendations. The goal is a figure that drops into a paper or a talk without
retouching: paper-matched fonts, no chartjunk, and lines heavy enough to survive
a projector.

Every rule below has a one-line rationale. When a figure genuinely needs to break
one, break it — but do it knowingly.

---

## 0. The preamble

The style lives in [schwingerModel/plotting.py](schwingerModel/plotting.py).
Do not paste it into notebooks:

```python
from schwingerModel import plotting as sp

sp.setStyle()          # house rcParams, globally
```

That gives you the palette (`sp.JLab_blue`, …), `sp.eb_kw`, `sp.LW`, `sp.GREY`,
`sp.BAND`, `sp.TEXTBOX`, and `sp.fmtErr`. The plot functions in that module call
`setStyle()` themselves on first use, so the explicit call is only needed when
you are hand-rolling a figure.

`setStyle` also silences fontTools' `'created' timestamp seems very low`
warnings. They come from the Computer Modern fonts matplotlib ships (`cmr10`,
`cmmi10`, `cmsy10`), whose `head`-table timestamps predate what fontTools
expects, so `mathtext.fontset: "cm"` emits two lines per font under every
figure. Cosmetic, and not a reason to give up CM math — pass
`setStyle(quietFonts=False)` to hear them again.

For reference, this is what `setStyle` applies and why:

```python
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


# Main JLab palette — first choice, in this order
JLab_red    = rgb_to_hex((192, 39, 45))
JLab_orange = rgb_to_hex((249, 102, 0))
JLab_blue   = rgb_to_hex((47, 122, 121))
JLab_green  = rgb_to_hex((65, 125, 10))

# Extended palette — only when the main four run out
bright_blue      = rgb_to_hex((0, 109, 219))
rose_pink        = rgb_to_hex((255, 109, 182))
lavender_violet  = rgb_to_hex((182, 109, 255))
burnt_orange     = rgb_to_hex((219, 109, 0))

plt.rcParams.update({
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
})

# white-filled markers, thick everything, no connecting line
eb_kw = dict(ms=8, mfc="white", mew=2.0, elinewidth=2.0, capsize=3.5,
             capthick=2.0, ls="none", zorder=3)

LW = 2.5          # standard curve linewidth
GREY = "0.8"      # reference lines (zero, thresholds, asymptotes)
BAND = "0.93"     # shaded regions (fit windows)
```

---

## 1. Color

**Use the palette above; nothing else.** The main four (`JLab_red`,
`JLab_orange`, `JLab_blue`, `JLab_green`) come first. Reach into the extended
palette only when a figure needs a fifth or later color.

Do not use matplotlib's default `C0…C9` cycle. It reads as "unstyled" and the
default blue/orange do not sit next to the JLab palette in a multi-panel paper.

Give each physical quantity a **fixed color across the whole project**. If the
pion is `JLab_blue` in the dispersion figure, it is `JLab_blue` in the finite-
volume figure too. A reader who learns the mapping once should not relearn it.

Pair each color with a **distinct marker shape** (`o`, `s`, `D`, `^`). Color
alone fails for the ~5% of readers with color-vision deficiency and for the
grayscale printout that someone will inevitably make.

```python
nxStyle = {4: (JLab_red, "o"), 8: (JLab_orange, "s"),
           16: (JLab_blue, "D"), 32: (JLab_green, "^")}
```

Grey is not a data color. `GREY` and `BAND` are reserved for reference lines and
shaded regions so that "grey means context, color means data" holds everywhere.

---

## 2. Type

Use **the LaTeX font from the papers** — serif body text with Computer Modern
math. `font.family: serif` + `mathtext.fontset: cm` gets you there without a
TeX install. If a figure needs to match a specific paper preamble exactly (e.g.
custom macros or a non-CM math font), switch to real TeX:

```python
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",   # + the paper's packages
})
```

**Figure text should appear as large as or larger than the paper's body text.**
The sizes above assume the figure is placed at roughly its natural width in a
single-column layout. If you shrink a figure in LaTeX, scale the fonts *up*
first — never let `\includegraphics[width=0.5\textwidth]` do the shrinking for
you, because it shrinks the type along with everything else.

Sanity check: print the draft page, hold it at arm's length. If the axis labels
are harder to read than the caption, they are too small.

All labels are math-mode where they name a quantity:

```python
ax.set_xlabel(r"$m_\pi^\infty L$")
ax.set_ylabel(r"$a m_\pi$")
```

Working figures may carry a `set_title` with the run parameters
(`β`, `m₀`, `N_x`, `N_t`) — that is how you avoid mixing up ensembles at the
bench. **Strip the title before the figure goes in a paper**; that information
belongs in the caption.

---

## 3. Lines

**Every line is thick — thicker than instinct suggests.** `lw=2.5` for curves,
`axes.linewidth=1.6` for spines and ticks. Figures get shrunk, projected, and
photographed off a screen; a 1.0-width line does not survive any of that.

**No dashed or dotted lines.** They read as tentative — "lines that lack purpose
in life." Distinguish curves by color and, if needed, by an inline text label
next to the curve. The one defensible exception is a line that *means*
"not physical" (an extrapolation beyond the data, a guide to the eye); if you use
one, say so explicitly in the caption.

---

## 4. Data with errors

**Points with errors → white-filled markers with thick bars.** The white fill is
what makes overlapping points readable; `mec` carries the color.

```python
ax.errorbar(x, y, yerr=err, marker="o", color=JLab_blue, mec=JLab_blue, **eb_kw)
```

**Functions with errors → `fill_between`, never a dense forest of error bars.**
Once the points are close enough that the caps touch, the error bars stop being
readable marks and become texture. A band with the central curve on top reads
instantly.

```python
ax.fill_between(x, lo, hi, color=JLab_blue, alpha=0.20, lw=0, zorder=1)
ax.plot(x, central, color=JLab_blue, lw=LW, zorder=2)
```

Use `alpha=0.20` and `lw=0` — the band should be visible but must never compete
with the markers drawn on top of it. Keep the band the same hue as its curve.

Note the asymmetric-error convention used throughout this repo: bootstrap error
rows are stored as `[high-central, central-low]`, while matplotlib's `yerr`
wants `[lower, upper]`. Reverse them, and clip negatives to zero (a negative
entry means the central value fell outside its own bootstrap band at noisy `t`):

```python
yerr = np.clip(err, 0.0, None)[::-1]
```

---

## 5. Zero, thresholds, and reference values

**Highlight zero and any threshold with a light grey line** — a two-particle
threshold, a fitted infinite-volume value, the ratio-equals-one line. The reader
should see the reference without it competing with the data.

```python
ax.axhline(0.0,  color=GREY, lw=LW, zorder=0)
ax.axhline(mInf, color=GREY, lw=LW, zorder=0)     # fitted asymptote
ax.axvspan(fitT[0], fitT[1], color=BAND, zorder=0)  # fit window
```

The alternative, for a threshold you want marked on the axis rather than across
the panel, is an **open circle tick** sitting on the spine:

```python
ax.plot([thresholdX], [ax.get_ylim()[0]], marker="o", ms=9, mfc="white",
        mec="0.4", mew=2.0, clip_on=False, zorder=4)
```

`clip_on=False` lets the marker straddle the axis line, which is the point — it
reads as an annotation of the axis, not as a data point.

---

## 6. Legends

**Do not use `plt.legend()`.** The default box — frame, sample line segments,
matplotlib's spacing — is the single most recognizable "this is an unstyled
Python plot" tell.

Instead, **build the legend as colored text**, each entry in the color of the
thing it names. It removes a whole layer of indirection: no sample swatch to
map back to a curve, and it gives the figure life.

```python
legendLines = [(r"$(aE_0)^2$", JLab_blue),
               (r"$(aE_1)^2$", JLab_green),
               (rf"cont: $c^2 = {fmtErr(c2, dc2)}$", JLab_red)]
for j, (txt, col) in enumerate(legendLines):
    ax.text(0.97, 0.05 + 0.075 * (len(legendLines) - 1 - j), txt, color=col,
            ha="right", va="bottom", transform=ax.transAxes, fontsize=15,
            zorder=5, bbox=sp.TEXTBOX)
```

The `bbox` (`sp.TEXTBOX`) is **fully transparent** by default. A solid white
patch reads as a box wherever it crosses a grey reference band or a shaded fit
window — `BAND` is only a few percent off white, so even a low alpha stays
visible as an edge. Place the block in whichever corner the data leaves empty
and it needs no backing at all; raise `sp.TEXTBOX["alpha"]` only for a label
that has to sit on dense markers.

For a figure that is going into a talk or a paper where it deserves real care,
**the final legend can be laid out by hand in Keynote** on top of the exported
PDF — that is the PI's habit and it is why `pdf.fonttype: 42` is set above (text
stays selectable and editable rather than being converted to outlines).

Numbers quoted in the legend use the parenthesized-error convention:

```python
def fmtErr(v, e):
    """0.7871(12)-style value with parenthesized error (2 sig figs)."""
    if not np.isfinite(v):
        return r"\mathrm{no\ fit}"
    if not np.isfinite(e) or e <= 0:
        return f"{v:.4f}"
    ndig = max(-int(np.floor(np.log10(e))) + 1, 0)
    return f"{v:.{ndig}f}({round(e * 10**ndig):d})"
```

---

## 7. Axis labels

Centered labels are the default and are fine. When a panel is wide, or when a
centered label would collide with the data or with a multi-panel neighbor,
**move the x-label to the far right** (and the y-label to the top):

```python
ax.set_xlabel(r"$(t - t_i)/a$", loc="right")
ax.set_ylabel(r"$a m_\pi$", loc="top", rotation=0, labelpad=-10)
```

This works especially well for time-series panels, where the far-right label
sits at the end of the axis it describes and frees the space under the panel.

In a row of panels sharing an axis, label it **once** — on the leftmost panel
for `y`, on the row for `x`. Repeating an identical label in every panel is
noise.

---

## 8. Layering

Use explicit `zorder` so elements never fight over what sits on top:

| `zorder` | element                                     |
|----------|---------------------------------------------|
| 0        | reference lines (`axhline`), shaded windows |
| 1        | `fill_between` error bands                  |
| 2        | fitted / theory curves                      |
| 3        | data error bars and markers                 |
| 4        | axis annotations (open-circle threshold ticks) |
| 5        | hand-made legend and inline text            |

Data on top of theory, always. The measurement is the point of the figure.

---

## 9. Figure size, layout, and output

```python
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
```

- `layout="constrained"` — never `tight_layout()`, and never hand-tuned
  `subplots_adjust`. Constrained layout handles the far-right labels and
  multi-panel spacing correctly.
- Single panel: `figsize=(8, 5.5)`, or `(7, 5.5)` for a squarer plot.
- A row of `n` panels: `figsize=(4 * n, 4.2)`, with
  `fig.get_layout_engine().set(w_pad=0.12)`.
- Set axis limits explicitly when late-time noise or an outlier would otherwise
  drive the autoscale. Leave headroom for the legend block.

**Save vector PDF into [figs/](figs/)**, never PNG — PDF stays sharp at any zoom
and remains editable in Keynote.

```python
fig.savefig(f"figs/pionDispersion_beta{meta.beta}_Nx{meta.dimx}_Nt{meta.dimt}.pdf")
```

Filenames are `camelCaseQuantity_param{value}_param{value}.pdf`, carrying every
parameter needed to identify the ensemble. A figure you cannot trace back to its
run is a figure you cannot put in a paper.

---

## 10. Ready-made plots

[schwingerModel/plotting.py](schwingerModel/plotting.py) has plot functions that
consume `bootstrapEnsemble` output verbatim — the `[central, err, cov]` triple —
so notebooks never unpack bootstrap internals or re-derive the error-row
convention.

```python
from schwingerModel import plotting as sp

prinCorrels = [sim.GEVP.bootstrapEnsemble(m, reduce=sim.GEVP.makeGevpReduce(ti=2, shift=1))
               for m in measuredPerVolume]
masses      = [sim.GEVP.bootstrapEnsemble(m, reduce=sim.GEVP.massReduce(ti=2, shift=1,
                                                                       fitT=(2, 6), withAmp=True))
               for m in measuredPerVolume]

sp.prinCorrelOverFit(prinCorrels, masses,
                     labels=[rf"$N_x = {nx}$" for nx in (4, 8, 16, 32)],
                     state=0, fitT=(2, 6), tmax=13,
                     savePath="figs/pionPrinCorrelOverFit_beta4_Nx4-32.pdf")
```

`massReduce` must be built with `withAmp=True` — the amplitude is what sets the
curve being divided out; the function raises if it isn't.

Panels wrap at **4 per row**, since a wider row shrinks each panel below
single-column paper width, and wrapping costs less vertical space than one panel
per row. `sp.makeGrid(n)` exposes that layout for new plot functions: it returns
`(fig, axes, nRows, nCols)` with a flat length-`n` axes array and any ragged
trailing axes already hidden. Label the x axis on panels with nothing below them
(`i + nCols >= n`) and the y axis on the leftmost of each row (`i % nCols == 0`).

---

## 11. Checklist

Before a figure leaves the notebook:

- [ ] Top and right spines off
- [ ] Serif / CM fonts; text as large as the paper's body text
- [ ] Palette colors only; each series has its own marker shape
- [ ] Every line `lw >= 2.5`; no dashed or dotted lines
- [ ] Error bar markers white-filled with thick bars
- [ ] Continuous error → `fill_between`, not a wall of error bars
- [ ] Zero and any threshold marked in light grey
- [ ] No `plt.legend()` — colored text block instead
- [ ] `zorder` set: data above curves above references
- [ ] Axis limits chosen, not autoscaled into noise
- [ ] Saved as PDF in `figs/` with the ensemble parameters in the name
- [ ] Working title stripped if the figure is paper-bound
