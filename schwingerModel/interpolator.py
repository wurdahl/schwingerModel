"""
Symbolic interpolator layer for automatic Wick contractions.

Layer 1 of the contraction pipeline: build interpolating operators (sums of
products of quark bilinears) from quantum numbers via isospin Clebsch-Gordan
coupling and symmetry projection. Purely symbolic — no gauge fields, no numerics.
The Wick contraction engine that consumes these lives in wick.py.

Central objects:
  MesonOp       (gamma, DNum, momk) — the payload shared with the numeric layer
  Bilinear      psibar_aq Gamma psi_q with a MesonOp payload
  Interpolator  complex-linear combination of products of Bilinears
  IsoMultiplet  irreducible isospin tensor: {I3: Interpolator}, Condon-Shortley phases

The only hand-written representation theory is the elementary meson multiplet
(quark ⊗ antiquark → triplet/singlet); everything else is the general coupler
plus symmetry projectors. Public isospins (multiplets, coupling targets) are
plain ints; half-integers appear only inside the elementary builder.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

TOL = 1e-12


class MesonOp(NamedTuple):
    """Bilinear structure label: gamma matrix, covariant-derivative count, momentum.
    Used symbolically here and as the elemental/gamma lookup key in distillation."""
    gamma: str
    DNum: int = 0
    momk: int = 0

# spatial-parity and Euclidean-time-reflection phases of the gamma structures
# (each covariant derivative contributes an extra (-1) to parity, nothing to eta_T)
GAMMA_PARITY  = {"id": +1, "g5": -1, "gx": -1, "gt": +1}
GAMMA_TIMEREV = {"id": +1, "g5": +1, "gx": -1, "gt": -1}


def clebschGordan(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M> as a float, via sympy (Condon-Shortley convention).
    Arguments may be ints or Fractions."""
    from sympy import Rational                      # lazy: keep package import fast
    from sympy.physics.wigner import clebsch_gordan
    args = [Rational(x) for x in (j1, j2, J, m1, m2, M)]
    return float(clebsch_gordan(*args))


class Bilinear(NamedTuple):
    aq: str            # flavor of psibar
    q: str             # flavor of psi
    op: MesonOp        # (gamma, DNum, momk) — numeric payload, reused by evaluation
    bar: bool = False  # True: use Gammabar = gammat Gamma^dag gammat (daggered source)

    def __str__(self):
        tags = []
        if self.op.DNum: tags.append(f"D{self.op.DNum}")
        if self.op.momk: tags.append(f"k={self.op.momk:+d}")
        tag = "," + ",".join(tags) if tags else ""
        barMark = "~" if self.bar else ""
        return f"{self.aq}_bar {self.op.gamma}{barMark}{tag} {self.q}"


def _canonical(monomial):
    """Bilinears are Grassmann-even, so products commute: canonical order = sorted."""
    return tuple(sorted(monomial))


class Interpolator:
    """Complex-linear combination of monomials (products of Bilinears)."""

    def __init__(self, terms=None):
        self._terms = {}                       # canonical monomial -> complex coeff
        if terms:
            for mono, coeff in terms.items():
                self._addTerm(mono, coeff)

    # -- construction ------------------------------------------------------
    @staticmethod
    def fromBilinears(coeff, *bils):
        out = Interpolator()
        out._addTerm(tuple(bils), coeff)
        return out

    def _addTerm(self, monomial, coeff):
        key = _canonical(monomial)
        newCoeff = self._terms.get(key, 0) + complex(coeff)
        if abs(newCoeff) < TOL:
            self._terms.pop(key, None)
        else:
            self._terms[key] = newCoeff

    # -- algebra -----------------------------------------------------------
    def __add__(self, other):
        out = Interpolator(self._terms)
        for mono, coeff in other._terms.items():
            out._addTerm(mono, coeff)
        return out

    def __sub__(self, other):
        return self + (-1) * other

    def __neg__(self):
        return (-1) * self

    def __mul__(self, other):
        if isinstance(other, Interpolator):    # tensor product: concatenate monomials
            out = Interpolator()
            for m1, c1 in self._terms.items():
                for m2, c2 in other._terms.items():
                    out._addTerm(m1 + m2, c1 * c2)
            return out
        return self.__rmul__(other)

    def __rmul__(self, scalar):
        out = Interpolator()
        for mono, coeff in self._terms.items():
            out._addTerm(mono, scalar * coeff)
        return out

    def dagger(self):
        """(psibar_a Gamma psi_b)^dag = psibar_b Gammabar psi_a: swap flavors, flip bar."""
        out = Interpolator()
        for mono, coeff in self._terms.items():
            newMono = tuple(Bilinear(b.q, b.aq, b.op, not b.bar) for b in mono)
            out._addTerm(newMono, coeff.conjugate())
        return out

    # -- inspection --------------------------------------------------------
    def terms(self):
        return dict(self._terms)

    def isZero(self):
        return not self._terms

    def norm(self):
        return math.sqrt(sum(abs(c) ** 2 for c in self._terms.values()))

    def normalized(self):
        n = self.norm()
        if n < TOL:
            raise ValueError("cannot normalize the zero interpolator")
        return (1.0 / n) * self

    def __eq__(self, other):
        if not isinstance(other, Interpolator):
            return NotImplemented
        keys = set(self._terms) | set(other._terms)
        return all(abs(self._terms.get(k, 0) - other._terms.get(k, 0)) < 1e-9
                   for k in keys)

    def __repr__(self):
        if self.isZero():
            return "Interpolator(0)"
        parts = []
        for mono, coeff in sorted(self._terms.items()):
            monoStr = " · ".join(str(b) for b in mono)
            parts.append(f"({coeff:+.4g})[{monoStr}]")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Isospin multiplets
# ---------------------------------------------------------------------------

class IsoMultiplet:
    """Irreducible isospin-I tensor operator: members[I3] is an Interpolator.
    I and the member keys are plain ints."""

    def __init__(self, I, members):
        self.I = I
        self.members = dict(members)

    def __getitem__(self, I3):
        return self.members[I3]

    def I3Values(self):
        return sorted(self.members)


_HALF = Fraction(1, 2)
# quark doublet: m=+1/2 -> u, m=-1/2 -> d
_QUARK = {+_HALF: ("u", 1), -_HALF: ("d", 1)}
# antiquark doublet in Condon-Shortley convention: m=+1/2 -> -dbar, m=-1/2 -> +ubar
_ANTIQUARK = {+_HALF: ("d", -1), -_HALF: ("u", 1)}


def mesonMultiplet(op: MesonOp, I):
    """
    Elementary quark-bilinear multiplet psibar Gamma psi coupled to isospin I (0 or 1).
    The only place half-integer isospins appear; everything larger goes through
    couple().
    """
    if I not in (0, 1):
        raise ValueError("a single quark bilinear carries isospin 0 or 1")
    members = {}
    for I3 in range(-I, I + 1):
        acc = Interpolator()
        for mA, (aqFlav, aqPhase) in _ANTIQUARK.items():
            mQ = I3 - mA
            if mQ in _QUARK:
                qFlav, qPhase = _QUARK[mQ]
                cg = clebschGordan(_HALF, mA, _HALF, mQ, I, I3)
                if cg != 0:
                    acc = acc + Interpolator.fromBilinears(
                        cg * aqPhase * qPhase, Bilinear(aqFlav, qFlav, op))
        members[I3] = acc
    return IsoMultiplet(I, members)


def couple(A: IsoMultiplet, B: IsoMultiplet, I):
    """General CG coupling of two multiplets to total isospin I."""
    if not abs(A.I - B.I) <= I <= A.I + B.I:
        raise ValueError(f"cannot couple I={A.I} and I={B.I} to I={I}")
    members = {}
    for I3 in range(-I, I + 1):
        acc = Interpolator()
        for mA, intA in A.members.items():
            mB = I3 - mA
            if mB in B.members:
                cg = clebschGordan(A.I, mA, B.I, mB, I, I3)
                if cg != 0:
                    acc = acc + cg * (intA * B.members[mB])
        members[I3] = acc
    return IsoMultiplet(I, members)


# ---------------------------------------------------------------------------
# Symmetry transforms and projectors
# ---------------------------------------------------------------------------

def parityTransform(interp: Interpolator):
    """Spatial parity: momk -> -momk, sign from gamma structure and derivative count."""
    out = Interpolator()
    for mono, coeff in interp.terms().items():
        sign = 1
        newMono = []
        for b in mono:
            sign *= GAMMA_PARITY[b.op.gamma] * (-1) ** b.op.DNum
            newMono.append(Bilinear(b.aq, b.q,
                                    MesonOp(b.op.gamma, b.op.DNum, -b.op.momk), b.bar))
        out._addTerm(tuple(newMono), sign * coeff)
    return out


def project(interp: Interpolator, transform, eigenvalue):
    """Project onto the +/-1 eigenspace of an involutive transform."""
    return 0.5 * (interp + eigenvalue * transform(interp))


def totalMomentum(interp: Interpolator):
    """Set of total momenta appearing across monomials (a good operator has one)."""
    return {sum(b.op.momk for b in mono) for mono in interp.terms()}


def parityEigenvalue(interp: Interpolator):
    """+1/-1 if the interpolator is a parity eigenstate, else None."""
    if interp.isZero():
        return None
    for s in (+1, -1):
        if parityTransform(interp) == s * interp:
            return s
    return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def makeState(content, I, I3, P=None, momTotal=0, intermediates=None):
    """
    Build an interpolator from particle content and quantum numbers.

    content:       list of (MesonOp, I_i) — each constituent's bilinear structure
                   (momentum lives inside the MesonOp) and its isospin
    I, I3:         total isospin and third component (ints)
    P:             if given, project onto that spatial-parity eigenvalue (+1/-1)
    momTotal:      required total momentum (validated)
    intermediates: for 3+ constituents, the intermediate isospins of the
                   left-fold coupling ((c1 c2) c3 ...), length len(content)-2

    Returns the (possibly zero) Interpolator. A zero result means the requested
    quantum numbers are impossible for this content — that is physics, not error.
    """
    mults = [mesonMultiplet(op, Ii) for op, Ii in content]

    cur = mults[0]
    if len(mults) == 1:
        if cur.I != I:
            raise ValueError(f"single bilinear has I={cur.I}, requested I={I}")
    else:
        mids = list(intermediates) if intermediates is not None else []
        if len(mults) > 2 and len(mids) != len(mults) - 2:
            raise ValueError("need len(content)-2 intermediate isospins")
        for k, nxt in enumerate(mults[1:]):
            target = I if k == len(mults) - 2 else mids[k]
            cur = couple(cur, nxt, target)

    interp = cur[I3]

    if P is not None:
        interp = project(interp, parityTransform, P)

    if not interp.isZero():
        moms = totalMomentum(interp)
        if moms != {momTotal}:
            raise ValueError(f"total momentum {moms} != required {momTotal}")

    return interp
