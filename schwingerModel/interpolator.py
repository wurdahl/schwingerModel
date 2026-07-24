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
    """Clebsch-Gordan coefficient <j1 m1; j2 m2 | J M> via sympy.

    Args:
        j1: Isospin of the first factor (int or Fraction).
        m1: Third component of the first factor.
        j2: Isospin of the second factor.
        m2: Third component of the second factor.
        J: Total isospin of the coupled state.
        M: Third component of the coupled state.

    Returns:
        float: The coefficient in the Condon-Shortley convention.
    """
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
        """Build a single-monomial Interpolator.

        Args:
            coeff: Complex coefficient of the monomial.
            *bils: Bilinears whose product forms the monomial.

        Returns:
            Interpolator: The one-term interpolator coeff * bils[0] * bils[1] * ...
        """
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
        """Hermitian conjugate: (psibar_a Gamma psi_b)^dag = psibar_b Gammabar psi_a.

        Returns:
            Interpolator: New interpolator with flavors swapped, bar flags
            flipped, and coefficients conjugated. self is unchanged.
        """
        out = Interpolator()
        for mono, coeff in self._terms.items():
            newMono = tuple(Bilinear(b.q, b.aq, b.op, not b.bar) for b in mono)
            out._addTerm(newMono, coeff.conjugate())
        return out

    # -- inspection --------------------------------------------------------
    def terms(self):
        """Copy of the term dictionary.

        Returns:
            dict[tuple[Bilinear, ...], complex]: Canonical (sorted) monomial ->
            coefficient.
        """
        return dict(self._terms)

    def isZero(self):
        """Returns:
            bool: True if the interpolator has no terms.
        """
        return not self._terms

    def norm(self):
        """Returns:
            float: L2 norm of the coefficient vector, sqrt(sum |c|^2).
        """
        return math.sqrt(sum(abs(c) ** 2 for c in self._terms.values()))

    def normalized(self):
        """Unit-norm copy of this interpolator.

        Returns:
            Interpolator: self / self.norm().

        Raises:
            ValueError: If the interpolator is zero.
        """
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
    I and the member keys are plain ints.

    Attributes:
        I: Total isospin (int).
        members: dict I3 (int) -> Interpolator.
    """

    def __init__(self, I, members):
        self.I = I
        self.members = dict(members)

    def __getitem__(self, I3):
        """Args:
            I3: Third component of isospin (int).

        Returns:
            Interpolator: The member with that I3.
        """
        return self.members[I3]

    def I3Values(self):
        """Returns:
            list[int]: The available I3 values, ascending.
        """
        return sorted(self.members)


_HALF = Fraction(1, 2)
# quark doublet: m=+1/2 -> u, m=-1/2 -> d
_QUARK = {+_HALF: ("u", 1), -_HALF: ("d", 1)}
# antiquark doublet in Condon-Shortley convention: m=+1/2 -> -dbar, m=-1/2 -> +ubar
_ANTIQUARK = {+_HALF: ("d", -1), -_HALF: ("u", 1)}


def mesonMultiplet(op: MesonOp, I):
    """Elementary quark-bilinear multiplet psibar Gamma psi coupled to isospin I.

    The only place half-integer isospins appear; everything larger goes through
    couple().

    Args:
        op: MesonOp instance carried by every Bilinear — its (gamma, DNum, momk)
            payload. Note: an instance, not the MesonOp class itself.
        I: Total isospin, 0 (singlet) or 1 (triplet).

    Returns:
        IsoMultiplet: Multiplet whose members carry Condon-Shortley phases
        (e.g. pi+ = -dbar Gamma u).

    Raises:
        ValueError: If I is not 0 or 1.
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
    """General Clebsch-Gordan coupling of two multiplets.

    Args:
        A: First IsoMultiplet.
        B: Second IsoMultiplet.
        I: Target total isospin (int); must satisfy |A.I - B.I| <= I <= A.I + B.I.

    Returns:
        IsoMultiplet: The coupled isospin-I multiplet; each member is a sum of
        tensor products A[mA] * B[I3 - mA] weighted by CG coefficients.

    Raises:
        ValueError: If the triangle inequality is violated.
    """
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
    """Apply the spatial-parity transform to an interpolator.

    Args:
        interp: Interpolator to transform.

    Returns:
        Interpolator: New interpolator with momk -> -momk in every MesonOp and
        each monomial multiplied by the product of GAMMA_PARITY[gamma] *
        (-1)^DNum over its bilinears.
    """
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
    """Project onto the +/-1 eigenspace of an involutive transform.

    Args:
        interp: Interpolator to project.
        transform: Involutive map Interpolator -> Interpolator
            (e.g. parityTransform).
        eigenvalue: Target eigenvalue, +1 or -1.

    Returns:
        Interpolator: (interp + eigenvalue * transform(interp)) / 2; zero if
        interp has no component in that eigenspace.
    """
    return 0.5 * (interp + eigenvalue * transform(interp))


def totalMomentum(interp: Interpolator):
    """Total momenta appearing across an interpolator's monomials.

    Args:
        interp: Interpolator to inspect.

    Returns:
        set[int]: Sum of momk over the bilinears of each monomial; a good
        momentum eigenstate yields a single-element set.
    """
    return {sum(b.op.momk for b in mono) for mono in interp.terms()}


def parityEigenvalue(interp: Interpolator):
    """Spatial-parity eigenvalue of an interpolator, if it has one.

    Args:
        interp: Interpolator to test.

    Returns:
        int | None: +1 or -1 if interp is a parity eigenstate; None if it is
        zero or not an eigenstate.
    """
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
    """Build an interpolator from particle content and quantum numbers.

    Args:
        content: List of (MesonOp, I_i) pairs — each constituent's bilinear
            structure (momentum lives inside the MesonOp) and its isospin.
        I: Total isospin (int).
        I3: Third component of total isospin (int).
        P: If given, project onto that spatial-parity eigenvalue (+1 or -1).
            Defaults to None (no projection).
        momTotal: Required total momentum; validated against the built
            interpolator. Defaults to 0.
        intermediates: For 3+ constituents, the intermediate isospins of the
            left-fold coupling ((c1 c2) c3 ...), length len(content) - 2.
            Defaults to None.

    Returns:
        Interpolator: The (possibly zero) interpolator. A zero result means the
        requested quantum numbers are impossible for this content — that is
        physics, not error.

    Raises:
        ValueError: If the isospin coupling chain is inconsistent, the number of
            intermediates is wrong, or the total momentum does not match momTotal.
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
