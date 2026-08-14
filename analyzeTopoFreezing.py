"""Integrated autocorrelation time of the topological charge vs beta.

Reads the eight scan_beta_m0.2.sh ensembles (32x32, m=0.2, N5=16, beta 3 -> 100),
computes Q per configuration, and estimates tau_int(Q) by Madras-Sokal windowing
*within* chains, then plots tau_int vs beta on a log axis with a tunnelling
diagnostic panel underneath.

Two things drive the structure here:

  * Configurations are stored CHAIN-MAJOR. experiment_gpu.runExperimentDwf
    transposes the (steps, chains, ...) history to (chains, steps, ...) and then
    concatenates, so links[0:nSteps] is chain 0, links[nSteps:2*nSteps] is chain
    1, and so on. Reshaping to (nChains, nSteps) therefore recovers the chains
    exactly; autocorrelations are computed per chain and never across a chain
    boundary, where the Markov history is discontinuous.

  * At large beta the chains stop tunnelling. Once Q is constant over a whole
    chain there is no decorrelation to measure and tau_int is only bounded from
    below, so those points are reported and drawn as lower limits rather than as
    measurements. The windowing criterion detects this on its own: a chain (or a
    set of chains) frozen at fixed Q has rho(t) = 1 for every lag, so W >= c*tau
    is never satisfied and the estimate runs away with the window.

Run from this directory inside the 'science' conda env:
    python analyzeTopoFreezing.py
"""

from __future__ import annotations

import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt

from schwingerModel import plotting as sp
from schwingerModel import topology

BETAS = ["3.0", "4.95", "8.17", "13.48", "22.24", "36.69", "60.54", "100.0"]
FMASS = "0.2"
NX, NT, N5 = 32, 32, 16

CONFIG_DIR = "configs"
LOG_DIR = "logs"
FIG_PATH = "figs/topoFreezingTauInt_beta3-100_m0.2_Nx32_Nt32.pdf"

#Madras-Sokal window factor: W is the smallest lag with W >= C_WINDOW*tau_int(W).
#5 is the usual choice for an exponential-ish rho; larger c is safer for a slowly
#decaying tail but costs statistical precision.
C_WINDOW = 5.0

CHUNK = 2000            # configs per read; one chunk of 32x32 complex is ~33 MB


def ensemblePath(beta):
    return f"{CONFIG_DIR}/dwf_beta_{beta}_m_{FMASS}_Nx_{NX}_Nt_{NT}_N5_{N5}.h5"


def logPath(beta):
    return f"{LOG_DIR}/scan_beta_m{FMASS}_b{beta}.log"


def chainStructure(beta, nConfigs, default=16):
    """(nChains, nSteps) for this ensemble.

    nChains is not stored in the hdf5 attrs, so it is read back from the run log
    line "running 16 chains x (300 burn-in + 1250 configs)" and cross-checked
    against the config count. Falls back to `default` if the log is unavailable.
    """
    nChains = default
    try:
        with open(logPath(beta)) as fh:
            m = re.search(r"running\s+(\d+)\s+chains\s+x\s+\((\d+) burn-in \+ (\d+) configs\)",
                          fh.read())
        if m:
            nChains, nSteps = int(m.group(1)), int(m.group(3))
            if nChains * nSteps != nConfigs:
                raise ValueError(f"beta={beta}: log says {nChains}x{nSteps} but the "
                                 f"ensemble holds {nConfigs} configs")
            return nChains, nSteps
    except FileNotFoundError:
        pass
    if nConfigs % nChains:
        raise ValueError(f"beta={beta}: {nConfigs} configs is not divisible by {nChains}")
    return nChains, nConfigs // nChains


def topoCharges(path, chunk=CHUNK):
    """Q for every configuration in `path`, vectorised over the config axis.

    Same plaquette and the same (1/2pi) sum(arg) as schwingerModel.topology.getTopoQ
    -- that function takes one config at a time, which is far too slow for 20k of
    them; correctness against it is asserted in `_checkAgainstReference`.

    Returns:
        (Q, maxDev): rounded integer charges, and the largest distance any raw
        (unrounded) value sat from an integer -- a sanity check that the sum
        really is quantised.
    """
    with h5py.File(path, "r") as f:
        links = f["links"]
        n = links.shape[0]
        Q = np.empty(n)
        maxDev = 0.0
        for lo in range(0, n, chunk):
            block = links[lo:lo + chunk]                    # (m, dimx, dimt, 2)
            Ut, Ux = block[..., 0], block[..., 1]
            UtShiftX = np.roll(Ut, -1, axis=1)              # U_t(x+1, t)
            UxShiftT = np.roll(Ux, -1, axis=2)              # U_x(x, t+1)
            plaq = Ux * UtShiftX * np.conjugate(UxShiftT) * np.conjugate(Ut)
            raw = np.sum(np.angle(plaq), axis=(1, 2)) / (2 * np.pi)
            Q[lo:lo + chunk] = np.round(raw)
            maxDev = max(maxDev, float(np.max(np.abs(raw - np.round(raw)))))
    return Q, maxDev


def _checkAgainstReference(path, Q, nSample=5, seed=0):
    """Assert the vectorised Q matches topology.getTopoQ on a few configs."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        n = f["links"].shape[0]
        idx = np.sort(rng.choice(n, size=min(nSample, n), replace=False))
        for i in idx:
            ref = topology.getTopoQ(f["links"][i])
            if ref != Q[i]:
                raise AssertionError(f"{path} cfg{i}: vectorised Q={Q[i]} != "
                                     f"topology.getTopoQ={ref}")
    return len(idx)


def autocorrFunction(series, maxLag):
    """Normalised rho(t) pooled over replicas (chains).

    series is (nChains, nSteps). Gamma(t) is averaged over chains about the
    GLOBAL mean, which is the standard replica treatment: each chain samples the
    same distribution, so pooling the autocovariance uses all the data while
    still never pairing configs across a chain boundary.

    Returns:
        (rho, gamma0): rho[0] == 1 by construction; gamma0 is the pooled
        variance, which is exactly 0 when every chain sits at one common Q.
    """
    nChains, nSteps = series.shape
    dev = series - series.mean()
    gamma = np.empty(maxLag + 1)
    for t in range(maxLag + 1):
        # pair i with i+t inside each chain only
        gamma[t] = np.mean([np.dot(dev[r, :nSteps - t], dev[r, t:]) / (nSteps - t)
                            for r in range(nChains)])
    if gamma[0] <= 0:
        return None, 0.0
    return gamma / gamma[0], gamma[0]


def tauIntMadrasSokal(series, c=C_WINDOW, maxLag=None):
    """Madras-Sokal tau_int with automatic windowing.

    tau_int(W) = 1/2 + sum_{t=1..W} rho(t), with W the smallest lag satisfying
    W >= c*tau_int(W). The Madras-Sokal error is
    delta(tau) = tau * sqrt(2*(2W+1)/N) with N the total number of configs.

    Returns a dict with tau, err, window, isLimit. isLimit is True when the
    window criterion is never met (the chains have not decorrelated within the
    lags available) or the pooled variance vanishes; tau is then a LOWER BOUND,
    not a measurement.
    """
    nChains, nSteps = series.shape
    nTotal = nChains * nSteps
    if maxLag is None:
        maxLag = nSteps // 2            # beyond N/2 the covariance estimate is junk

    rho, gamma0 = autocorrFunction(series, maxLag)
    if rho is None:                     # every chain frozen at one common Q
        return dict(tau=nSteps / 2.0, err=np.nan, window=maxLag,
                    isLimit=True, gamma0=0.0)

    tauRunning = 0.5 + np.cumsum(rho[1:])          # tauRunning[W-1] = tau_int(W)
    window = None
    for W in range(1, maxLag + 1):
        if W >= c * tauRunning[W - 1]:
            window = W
            break

    if window is None:                  # never decorrelated -> lower bound only
        tau = float(tauRunning[maxLag - 1])
        return dict(tau=tau, err=np.nan, window=maxLag, isLimit=True, gamma0=gamma0)

    tau = float(tauRunning[window - 1])
    err = tau * np.sqrt(2.0 * (2 * window + 1) / nTotal)
    return dict(tau=tau, err=err, window=window, isLimit=False, gamma0=gamma0)


def tunnellingStats(series):
    """Per-chain tunnelling diagnostics.

    Returns dict with nTunnelChains, tunnelFrac, jumpsPerChain (mean number of
    Q changes) and nDistinctQ across the whole ensemble. Deliberately does not
    return nChains: analyse() supplies that alongside, and duplicating it here
    collides when both dicts are splatted into one row.
    """
    jumps = np.sum(np.diff(series, axis=1) != 0, axis=1)
    return dict(nTunnelChains=int(np.sum(jumps > 0)),
                tunnelFrac=float(np.mean(jumps > 0)),
                jumpsPerChain=float(np.mean(jumps)),
                nDistinctQ=int(np.unique(series).size))


def analyse():
    rows = []
    for beta in BETAS:
        path = ensemblePath(beta)
        if not os.path.exists(path):
            print(f"  beta={beta}: MISSING {path} -- skipped")
            continue
        Q, maxDev = topoCharges(path)
        nChecked = _checkAgainstReference(path, Q)
        nChains, nSteps = chainStructure(beta, Q.size)
        series = Q.reshape(nChains, nSteps)     # chain-major: see module docstring

        tau = tauIntMadrasSokal(series)
        tun = tunnellingStats(series)
        rows.append(dict(beta=float(beta), betaStr=beta, Q=Q,
                         maxDev=maxDev, nChecked=nChecked,
                         nChains=nChains, nSteps=nSteps, **tau, **tun))

        limitTag = ">" if tau["isLimit"] else " "
        errTag = "     (lower limit)" if tau["isLimit"] else f" +- {tau['err']:.2f}"
        print(f"  beta={beta:<7} tau_int {limitTag}{tau['tau']:8.2f}{errTag}"
              f"   W={tau['window']:<4d} tunnelling chains "
              f"{tun['nTunnelChains']}/{nChains}"
              f"  jumps/chain {tun['jumpsPerChain']:7.1f}"
              f"  distinct Q {tun['nDistinctQ']}"
              f"  |maxdev {maxDev:.2e}|")
    return rows


def makeFigure(rows, path=FIG_PATH):
    sp.setStyle()
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(7.2, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))

    beta = np.array([r["beta"] for r in rows])
    tau = np.array([r["tau"] for r in rows])
    err = np.array([r["err"] for r in rows])
    isLim = np.array([r["isLimit"] for r in rows])

    # measurements: house style is a white-filled marker with visible caps
    if np.any(~isLim):
        ax.errorbar(beta[~isLim], tau[~isLim], yerr=err[~isLim],
                    marker="o", color=sp.JLab_blue, mec=sp.JLab_blue, **sp.eb_kw)
    # lower limits: open symbol + upward arrow, never a fake central value
    if np.any(isLim):
        lim_kw = {k: v for k, v in sp.eb_kw.items() if k != "capsize"}
        ax.errorbar(beta[isLim], tau[isLim],
                    yerr=0.35 * tau[isLim], lolims=True,
                    marker="o", color=sp.JLab_red, mec=sp.JLab_red,
                    capsize=0.0, **lim_kw)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\tau_\mathrm{int}(Q)$  [configs]")
    ax.set_title(r"Topological freezing: $32\times32$, $m=0.2$, $N_5=16$",
                 fontsize=18, pad=10)

    # headroom so the limit arrows are not clipped against the top spine
    finite = tau[np.isfinite(tau)]
    ax.set_ylim(0.4 * finite.min(), 4.0 * finite.max())

    # the chain length is the hard ceiling on what this statistics can resolve
    nSteps = rows[0]["nSteps"]
    ax.axhline(nSteps / 2.0, color=sp.GREY, lw=sp.LW, ls="--", zorder=0)
    ax.text(0.98, 0.86, rf"$N_\mathrm{{chain}}/2 = {nSteps // 2}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=14, color="0.45", bbox=sp.TEXTBOX)

    handles = []
    if np.any(~isLim):
        handles.append(plt.Line2D([], [], marker="o", color=sp.JLab_blue,
                                  mfc="white", mew=2.0, ls="none", ms=8,
                                  label="measured"))
    if np.any(isLim):
        handles.append(plt.Line2D([], [], marker="o", color=sp.JLab_red,
                                  mfc="white", mew=2.0, ls="none", ms=8,
                                  label="lower limit (frozen)"))
    # centre-left is the one genuinely empty region: the single measured point
    # sits bottom-left and every limit is pinned to the top of the panel
    ax.legend(handles=handles, frameon=False, fontsize=14, loc="center left")

    # the physics of the figure is where freezing switches on, so say it
    onset = [r for r in rows if r["tunnelFrac"] == 0.0]
    if onset:
        ax.axvspan(min(r["beta"] for r in onset), ax.get_xlim()[1],
                   color=sp.BAND, zorder=0)
        ax.text(0.985, 0.30, "no tunnelling\nin 1250 configs",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=13, color="0.45", bbox=sp.TEXTBOX)

    # diagnostic: what fraction of chains tunnelled at all
    frac = np.array([r["tunnelFrac"] for r in rows])
    axd.plot(beta, frac, marker="s", color=sp.JLab_orange, mfc="white",
             mew=2.0, ms=8, lw=sp.LW, zorder=3)
    axd.set_ylim(-0.08, 1.12)
    axd.set_xscale("log")
    axd.set_ylabel("tunnelling\nchains", fontsize=16)
    axd.set_xlabel(r"$\beta$")
    axd.axhline(1.0, color=sp.GREY, lw=sp.LW, zorder=0)
    axd.axhline(0.0, color=sp.GREY, lw=sp.LW, zorder=0)
    axd.set_yticks([0.0, 0.5, 1.0])
    axd.set_yticklabels(["0", r"$\frac{1}{2}$", "all"])

    ticks = [3, 10, 30, 100]
    axd.set_xticks(ticks)
    axd.set_xticklabels([str(t) for t in ticks])
    axd.xaxis.set_minor_formatter(plt.NullFormatter())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"\nwrote {path}")
    return fig


def main():
    print(f"tau_int(Q) vs beta -- {NX}x{NT}, m={FMASS}, N5={N5}, "
          f"Madras-Sokal windowing at c={C_WINDOW}\n")
    rows = analyse()
    if not rows:
        raise SystemExit("no ensembles found")
    makeFigure(rows)
    return rows


if __name__ == "__main__":
    main()
