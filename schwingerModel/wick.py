"""
Wick contraction engine (layer 2 of the contraction pipeline).

Consumes Interpolators from interpolator.py and produces diagram tables:

    contract(snkOp, srcOp) -> {DiagramKey: coefficient}

for the correlator <snkOp(snk) srcOp^dag(src)>. The source is daggered here —
callers pass both operators in "creation" form.

A DiagramKey is a sorted tuple of canonical cycles. Each cycle is a tuple of
(vertex, flavor) pairs walked in contraction order:

    vertex = (timeLabel, MesonOp, bar)   — which elemental/gamma to insert
    flavor = flavor of the propagator LEAVING this vertex (the psi it provides)

so a cycle [(v1, f1), (v2, f2)] means Tr[ Gamma_1 S_f1(t1,t2) Gamma_2 S_f2(t2,t1) ],
with the overall sign (-1)^{#cycles} already folded into the coefficient.
Cycles are rotation-minimal and sorted within a key, so symmetry-equivalent
contractions merge (and cancellations drop out) automatically.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

from .interpolator import Bilinear, Interpolator, MesonOp, TOL


def _cycleDecomp(sigma):
    """Decompose a permutation into its orbits (cycles).

    Args:
        sigma: Permutation as a dict mapping index -> index.

    Returns:
        list[list[int]]: The orbits of sigma, each starting from its smallest
        element, ordered by that smallest element.
    """
    seen, orbits = set(), []
    for start in sorted(sigma):
        if start in seen:
            continue
        orbit, i = [], start
        while i not in seen:
            seen.add(i)
            orbit.append(i)
            i = sigma[i]
        orbits.append(orbit)
    return orbits


def _canonicalCycle(seq):
    """Rotation-minimal form of a cycle.

    Args:
        seq: Cycle as a tuple of (vertex, flavor) pairs.

    Returns:
        tuple: The lexicographically smallest rotation of seq, so that
        symmetry-equivalent cycles compare equal.
    """
    return min(seq[r:] + seq[:r] for r in range(len(seq)))


def _contractMonomials(bils, labels):
    """All Wick contractions of one monomial (product of Bilinears).

    Args:
        bils: List of Bilinears forming one monomial (sink and source concatenated).
        labels: Time label for each bilinear, parallel to bils (e.g. "snk"/"src").

    Returns:
        dict[DiagramKey, int]: Map from canonical diagram key to the summed Wick
        sign (-1)^{#cycles}. Coefficients of the interpolator terms are applied
        by the caller. Empty dict if the flavor content cannot be contracted.
    """
    psi, psibar = defaultdict(list), defaultdict(list)
    for i, b in enumerate(bils):
        psi[b.q].append(i)
        psibar[b.aq].append(i)

    if set(psi) != set(psibar) or any(len(psi[f]) != len(psibar[f]) for f in psi):
        return {}

    verts = [(labels[i], b.op, b.bar) for i, b in enumerate(bils)]
    flavors = sorted(psi)

    out = defaultdict(int)
    for perms in itertools.product(*(itertools.permutations(psibar[f]) for f in flavors)):
        # sigma: psi of bilinear i is contracted with psibar of bilinear sigma[i]
        sigma = {}
        for f, perm in zip(flavors, perms):
            for i, j in zip(psi[f], perm):
                sigma[i] = j
        orbits = _cycleDecomp(sigma)
        key = tuple(sorted(
            _canonicalCycle(tuple((verts[i], bils[i].q) for i in orbit))
            for orbit in orbits))
        out[key] += (-1) ** len(orbits)
    return out


def contract(snkOp: Interpolator, srcOp: Interpolator, snkLabel="snk", srcLabel="src"):
    """Diagram table for the correlator <snkOp(snkLabel) srcOp^dag(srcLabel)>.

    Both operators are given in creation form; the source is daggered here
    (flavor swap, bar flip, coefficient conjugation).

    Args:
        snkOp: Sink Interpolator, in creation form.
        srcOp: Source Interpolator, in creation form (daggered internally).
        snkLabel: Time label attached to sink vertices. Defaults to "snk".
        srcLabel: Time label attached to source vertices. Defaults to "src".

    Returns:
        dict[DiagramKey, complex]: Diagram table mapping canonical cycle keys to
        coefficients (Wick sign included), with numerically-zero entries dropped.
        An empty table means the correlator vanishes identically (e.g.
        mismatched flavor content).
    """
    src = srcOp.dagger()
    table = defaultdict(complex)
    for mSnk, cSnk in snkOp.terms().items():
        for mSrc, cSrc in src.terms().items():
            bils = list(mSnk) + list(mSrc)
            labels = [snkLabel] * len(mSnk) + [srcLabel] * len(mSrc)
            for key, signSum in _contractMonomials(bils, labels).items():
                table[key] += cSnk * cSrc * signSum
    return {k: v for k, v in table.items() if abs(v) > TOL}


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------

def mergeFlavors(table, flavorClass=None):
    """Map propagator flavors to their degenerate class and re-merge the table.

    This is both a symbolic tool (isospin-equal operators give identical merged
    tables) and the evaluation dictionary (class label -> which tau to use).

    Args:
        table: Diagram table {DiagramKey: coeff} from contract().
        flavorClass: Callable flavor -> class label. Defaults to None, which
            sends every flavor to "q" (exact isospin degeneracy, one tau for
            all flavors).

    Returns:
        dict[DiagramKey, complex]: Re-merged table with flavors replaced by
        class labels; entries that cancel after merging are dropped.
    """
    if flavorClass is None:
        flavorClass = lambda f: "q"
    out = defaultdict(complex)
    for key, coeff in table.items():
        newKey = tuple(sorted(
            _canonicalCycle(tuple((v, flavorClass(f)) for v, f in cyc))
            for cyc in key))
        out[newKey] += coeff
    return {k: v for k, v in out.items() if abs(v) > TOL}


def tablesEqual(a, b, tol=1e-9):
    """Compare two diagram tables with a numerical tolerance on coefficients.

    Args:
        a: First diagram table {DiagramKey: coeff}.
        b: Second diagram table {DiagramKey: coeff}.
        tol: Maximum allowed |a[k] - b[k]| per key. Defaults to 1e-9.

    Returns:
        bool: True if every key's coefficients agree within tol (missing keys
        count as zero).
    """
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0) - b.get(k, 0)) < tol for k in keys)


def cycleTimeLabels(cycle):
    """Set of time labels a cycle touches.

    Args:
        cycle: One canonical cycle — a tuple of (vertex, flavor) pairs.

    Returns:
        set[str]: The time labels visited: {'snk', 'src'} = connected in time,
        a single label = a loop (needs ensemble-level vacuum subtraction).
    """
    return {v[0] for v, _ in cycle}


def splitConnected(table):
    """Partition a diagram table by time-connectedness.

    A diagram is connected iff every one of its cycles touches both time labels.

    Args:
        table: Diagram table {DiagramKey: coeff}.

    Returns:
        tuple[dict, dict]: (connected, disconnected) sub-tables, each
        {DiagramKey: coeff}; together they partition the input.
    """
    conn, disc = {}, {}
    for key, coeff in table.items():
        if all(len(cycleTimeLabels(c)) > 1 for c in key):
            conn[key] = coeff
        else:
            disc[key] = coeff
    return conn, disc


def formatTable(table):
    """Human-readable rendering of a diagram table.

    Args:
        table: Diagram table {DiagramKey: coeff}.

    Returns:
        str: One line per diagram, "coeff  Tr[...] x Tr[...]", with gamma,
        derivative, momentum, and flavor annotations on each propagator step.
    """
    if not table:
        return "(empty table)"
    lines = []
    for key, coeff in sorted(table.items()):
        cycs = []
        for cyc in key:
            steps = []
            for (label, op, bar), fl in cyc:
                tags = "".join([f",D{op.DNum}" if op.DNum else "",
                                f",k={op.momk:+d}" if op.momk else ""])
                steps.append(f"{label}:{op.gamma}{'~' if bar else ''}{tags} --{fl}-->")
            cycs.append("Tr[ " + " ".join(steps) + " ]")
        lines.append(f"{coeff:+.4g}  " + "  x  ".join(cycs))
    return "\n".join(lines)
