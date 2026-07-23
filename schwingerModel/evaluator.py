"""
Numeric evaluation of Wick diagram tables against a DistillWorkspace (layer 3).

This is the single evaluation path for all operators — single mesons,
multi-hadron interpolators, isosinglets — replacing the hand-written einsums
in distillation.py (kept there as regression oracles until deprecated).

Pipeline position:
    interpolator.py  (states)  ->  wick.py (diagram tables)  ->  HERE  ->  GEVP.py

A diagram key from wick.contract is a tuple of cycles; each cycle alternates
(timeLabel, MesonOp, bar) vertices with propagator flavors. evalCycle turns one
cycle into an array over its distinct time labels; evalTable combines cycles
into per-diagram products and returns:

    EvalResult.conn : (T,) complex — translation-averaged sum of diagrams whose
                      cycles all connect sink and source times
    EvalResult.disc : list of (coeff, A, B) — for factorized diagrams
                      (sink-only cycles) x (source-only cycles): A(t) and B(t)
                      per-config loop series. The ensemble-level combination
                      (1/T) sum_t A(t+dt) B(t) minus the <A><B> vacuum term
                      belongs in the statistics layer, NOT here — vacuum
                      subtraction needs ensemble means.

Flavors on cycle edges are currently all mapped to the workspace's single tau
(exact isospin degeneracy). When flavors become numerically distinct, this is
the one place a flavor -> tau lookup gets added.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .wick import contract, mergeFlavors


class EvalResult(NamedTuple):
    conn: np.ndarray   # (T,) complex
    disc: list         # [(coeff, A (T,), B (T,)), ...]


def _translationAvg(M):
    """C(dt) = (1/T) sum_t M[t+dt, t] for M indexed (t_snk, t_src)."""
    T = M.shape[0]
    return np.array([np.roll(M, -dt, axis=0).diagonal().mean() for dt in range(T)])


def evalCycle(ws, cycle, snkLabel="snk", srcLabel="src"):
    """
    Evaluate one cycle: Tr[ E_1(t_1) tau(t_1,t_2) E_2(t_2) tau(t_2,t_3) ... ]
    with E = spatial elemental (x) gamma, all pulled from the workspace.

    Returns (labels, array): labels is the tuple of time labels the cycle
    touches (sink first), and array has one T-axis per label.
    """
    tau = ws.tau                            # (T, T, N, 2, N, 2)

    labels = tuple(l for l in (snkLabel, srcLabel)
                   if any(v[0] == l for v, _ in cycle))
    nextAxis = 0
    def newAxis():
        nonlocal nextAxis
        nextAxis += 1
        return nextAxis - 1

    timeAxis = {l: newAxis() for l in labels}

    operands, vertexAxes = [], []
    for (label, op, bar), _flavor in cycle:
        r, c, s, u = newAxis(), newAxis(), newAxis(), newAxis()
        vertexAxes.append((r, c, s, u))
        operands += [ws.elemental(op, bar), [timeAxis[label], r, c],
                     ws.gamma(op, bar),     [s, u]]

    n = len(cycle)
    for k in range(n):
        kn = (k + 1) % n
        tK  = timeAxis[cycle[k][0][0]]
        tKn = timeAxis[cycle[kn][0][0]]
        _, cK, _, uK = vertexAxes[k]
        rN, _, sN, _ = vertexAxes[kn]
        # tau[t_k, t_{k+1}, vec_out_k, spin_out_k, vec_in_{k+1}, spin_in_{k+1}]
        operands += [tau, [tK, tKn, cK, uK, rN, sN]]

    out = [timeAxis[l] for l in labels]
    return labels, np.einsum(*operands, out, optimize=True)


def evalTable(ws, table, snkLabel="snk", srcLabel="src"):
    """
    Evaluate a full diagram table on one workspace. Diagram coefficients
    (including the Wick sign) come from the table; no extra signs here.
    """
    T = ws.eigVecs.shape[0]
    conn = np.zeros(T, dtype=complex)
    disc = []

    for key, coeff in table.items():
        cross, snkOnly, srcOnly = [], [], []
        for cyc in key:
            labels, arr = evalCycle(ws, cyc, snkLabel, srcLabel)
            if len(labels) == 2:
                cross.append(arr)
            elif labels == (snkLabel,):
                snkOnly.append(arr)
            else:
                srcOnly.append(arr)

        if cross:
            # Mixed diagrams (cross cycles times single-time loops) are ordinary
            # per-config products: only FULLY factorized diagrams take part in
            # the ensemble-level vacuum subtraction, so loop factors here just
            # multiply the (t_snk, t_src) map pointwise.
            M = cross[0].copy()
            for m in cross[1:]:
                M = M * m
            for a in snkOnly:
                M = M * a[:, None]
            for b in srcOnly:
                M = M * b[None, :]
            conn = conn + coeff * _translationAvg(M)
        else:
            if not (snkOnly and srcOnly):
                raise ValueError("disconnected diagram missing a sink or source factor")
            A = snkOnly[0].copy()
            for a in snkOnly[1:]:
                A = A * a
            B = srcOnly[0].copy()
            for b in srcOnly[1:]:
                B = B * b
            disc.append((coeff, A, B))

    return EvalResult(conn, disc)


def evalCorrelator(ws, snkOp, srcOp):
    """
    Contract two interpolators (creation form) and evaluate on one workspace.

    Flavors are merged into their degenerate class before evaluation: the
    workspace has a single tau, so this is exact — and it lets isospin
    cancellations (e.g. the pi0 disconnected pieces) drop out symbolically
    instead of numerically. For ensemble runs, do this once outside the config
    loop and call evalTable directly:

        table = mergeFlavors(contract(snkOp, srcOp))
        ... per config: evalTable(ws, table)
    """
    return evalTable(ws, mergeFlavors(contract(snkOp, srcOp)))
