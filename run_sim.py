"""
Gauge configuration generation driven by a TOML input file:

    python run_sim.py inputs/example.toml

Runs nChains independent HMC chains in parallel, discards the per-chain burn-in,
merges the thermalized configurations, and writes them to an HDF5 ensemble file.
The number of HMC substeps is either given explicitly (numSubSteps) or tuned
automatically from the integrator's energy error (see tuneSubSteps), then
confirmed on an independent seed. See inputs/example.toml for all recognized fields.
"""

import math
import os
import sys
import tomllib

from scipy.special import erfcinv
from tqdm import tqdm

from schwingerModel import experiment


def loadInput(path):
    with open(path, 'rb') as f:
        raw = tomllib.load(f)

    run = raw['run']

    cfg = {
        #required
        'outputFile': raw['outputFile'],
        'beta':  raw['physics']['beta'],
        'fMass': raw['physics']['fMass'],
        'dimx':  raw['lattice']['dimx'],
        'dimt':  raw['lattice']['dimt'],
        #total metropolis steps across all chains (older inputs spell this targetConfigs)
        'metroSteps': run.get('metroSteps', run.get('targetConfigs')),
        'burnIn':     run['burnIn'],
        #optional
        'aSpacing': raw['physics'].get('aSpacing', 1.0),
        #the force solves are metropolis-corrected so they may run loose; the
        #action solves enter dH directly and stay tight. Old inputs' single
        #cgRtol is honored as the force tolerance.
        'cgRtolForce':  run.get('cgRtolForce', run.get('cgRtol', 1e-5)),
        'cgRtolAction': run.get('cgRtolAction', 1e-10),
        'randSeed': run.get('randSeed', 0),
        'nCores':   run.get('nCores', os.cpu_count()),
        'tunneling': run.get('tunneling', False),
        #restart every force CG from zero: exactly reversible leapfrog, so
        #<exp(-dH)>=1 holds exactly, at ~40% more solve time. False (the
        #default, and how every ensemble before 2026-08 was generated)
        #warm-starts from the previous substep's solution.
        'coldStartForce': run.get('coldStartForce', False),
        #where the chains run: "cpu" (joblib workers, default) or "gpu" (vmapped jax)
        'backend': run.get('backend', 'cpu'),
        #sea quark discretization: "wilson" (default) or "dwf" (needs dim5, M5)
        'fermionAction': raw['physics'].get('fermionAction', 'wilson'),
        'dim5': raw['physics'].get('dim5'),
        'M5':   raw['physics'].get('M5'),
    }
    if cfg['metroSteps'] is None:
        raise KeyError(f"{path} sets neither run.metroSteps nor run.targetConfigs")
    if cfg['fermionAction'] == 'dwf' and (cfg['dim5'] is None or cfg['M5'] is None):
        raise KeyError(f"{path} sets fermionAction='dwf' but not physics.dim5 and physics.M5")
    if cfg['backend'] not in ('cpu', 'gpu'):
        raise KeyError(f"{path} sets run.backend={cfg['backend']!r}; expected 'cpu' or 'gpu'")
    if cfg['backend'] == 'gpu' and cfg['fermionAction'] != 'dwf':
        raise KeyError(f"{path} sets run.backend='gpu', which only implements fermionAction='dwf'")
    if cfg['backend'] == 'gpu' and cfg['tunneling']:
        raise KeyError(f"{path} sets run.backend='gpu', which does not implement tunneling")
    cfg['nChains'] = run.get('nChains', cfg['nCores'])

    sub = raw.get('substeps', {})
    cfg['numSubSteps']      = sub.get('numSubSteps')        # None -> tune automatically
    cfg['targetAcceptance'] = sub.get('targetAcceptance', 0.8)
    cfg['pilotSteps']       = sub.get('pilotSteps', 100)
    cfg['startSubSteps']    = sub.get('startSubSteps', 5)
    cfg['maxSubSteps']      = sub.get('maxSubSteps', 500)
    #pilots run at the production chain count by default: the acceptance is
    #per-chain so the width costs nothing in bias, and on the gpu a wider batch
    #is far cheaper than the equivalent extra pilot steps. cpu pilots are one chain.
    cfg['pilotChains']      = sub.get('pilotChains',
                                      cfg['nChains'] if cfg['backend'] == 'gpu' else 1)
    #re-measure the tuned count on a seed the search never saw; 0 disables
    cfg['confirmSteps']     = sub.get('confirmSteps', cfg['pilotSteps'])
    cfg['maxPilots']        = sub.get('maxPilots', 12)

    return cfg


def getRunner(backend):
    """The module the run routes through. The gpu module is imported lazily because
    importing jax initializes the gpu (and preallocates most of its memory), which
    a cpu run must never do."""
    if backend == 'gpu':
        from schwingerModel import experiment_gpu
        return experiment_gpu
    return experiment


def pilotAcceptance(cfg, numSubSteps, pilotSteps=None, seed=None):
    """Acceptance rate of a short pilot chain, measured on its second half
    so the cold-start transient does not bias the estimate."""
    #only the gpu runner batches chains; the cpu one takes no such argument
    extra = {'chains': cfg['pilotChains']} if cfg['backend'] == 'gpu' else {}

    accept, _ = getRunner(cfg['backend']).acceptanceFractions(
        beta=cfg['beta'], fMass=cfg['fMass'], aSpacing=cfg['aSpacing'],
        Nx=cfg['dimx'], Nt=cfg['dimt'],
        metroSteps=cfg['pilotSteps'] if pilotSteps is None else pilotSteps,
        numSubSteps=numSubSteps,
        tunneling=cfg['tunneling'],
        cgRtolForce=cfg['cgRtolForce'], cgRtolAction=cfg['cgRtolAction'],
        fermionAction=cfg['fermionAction'], dim5=cfg['dim5'], M5=cfg['M5'],
        coldStartForce=cfg['coldStartForce'],
        seed=cfg['randSeed'] if seed is None else seed,
        tqdmPosition=1,   #keep the pilot bar below the tuning report lines
        **extra,
    )
    return accept


def pilotSamples(cfg, pilotSteps=None):
    """Accept/reject draws behind one pilot's acceptance estimate.

    acceptanceFractions measures only the second half of the chain, times
    however many chains ran in lockstep. This sets the pilot's noise, which is
    what the model below weights by and what the confirmation is tested against.
    """
    steps = cfg['pilotSteps'] if pilotSteps is None else pilotSteps
    return max((steps - int(steps * 0.5)) * cfg['pilotChains'], 1)


#--- integrator error model -------------------------------------------------
#A symmetric 2nd-order integrator leaves an energy error <dH> = C * n^-p over a
#fixed trajectory length, with p = 4 in theory, and an exact HMC step accepts
#with P = erfc(sqrt(<dH>/2)). Inverting that turns every pilot into a
#measurement of <dH>, so the substep count that hits the target follows from the
#model instead of from a finite-difference slope. Fitted p over the ensembles
#already generated runs 2.9-4.3, so p is refitted rather than assumed once there
#is more than one pilot -- but shrunk toward theory, because a two-point fit at
#low acceptance can otherwise extrapolate badly.
_P_THEORY  = 4.0
_P_BOUNDS  = (2.0, 8.0)
_P_PRIORVAR = 1.0      # p ~ 4 +- 1
_MODEL_SIGMA = 0.15    # log-space model error, floors the fit weights


def impliedError(acceptance, guard=1e-6):
    """<dH> implied by an acceptance, inverting P = erfc(sqrt(<dH>/2))."""
    a = min(max(acceptance, guard), 1.0 - guard)
    return 2.0 * erfcinv(a) ** 2


def pilotStdErr(acceptance, nSamples):
    """Binomial standard error on a pilot acceptance."""
    return math.sqrt(max(acceptance * (1.0 - acceptance), 0.01) / max(nSamples, 1))


def _logErrorSigma(acceptance, nSamples):
    """1-sigma on log(<dH>) propagated from the pilot's acceptance error.

    d log<dH>/dP = -sqrt(pi) exp(x^2) / x with x = erfcinv(P), which grows
    steeply as P -> 1: a pilot at 0.99 constrains the model roughly 15x more
    weakly than one at 0.10, which is why the fit is weighted.
    """
    guard = 0.5 / max(nSamples, 1)
    a = min(max(acceptance, guard), 1.0 - guard)
    x = erfcinv(a)
    slope = math.sqrt(math.pi) * math.exp(x * x) / max(x, 1e-12)
    return math.hypot(slope * pilotStdErr(a, nSamples), _MODEL_SIGMA)


def fitErrorModel(results, nSamples):
    """(logC, p) for <dH> = C * n^-p, weighted least squares in log space.

    A single pilot pins logC at the theoretical p; two or more also fit p,
    shrunk toward _P_THEORY by its own fit variance and clamped to _P_BOUNDS so
    a noisy pair cannot produce a runaway extrapolation.
    """
    guard = 0.5 / max(nSamples, 1)
    pts = [(math.log(n),
            math.log(impliedError(a, guard)),
            1.0 / _logErrorSigma(a, nSamples) ** 2)
           for n, a in results.items()]

    if len(pts) == 1:
        x, y, _ = pts[0]
        return y + _P_THEORY * x, _P_THEORY

    sw  = sum(w for _, _, w in pts)
    mx  = sum(w * x for x, _, w in pts) / sw
    my  = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)

    if sxx <= 1e-9:                       # every pilot at the same count
        p = _P_THEORY
    else:
        pFit, pVar = -sxy / sxx, 1.0 / sxx
        p = (pFit / pVar + _P_THEORY / _P_PRIORVAR) / (1.0 / pVar + 1.0 / _P_PRIORVAR)
    p = min(max(p, _P_BOUNDS[0]), _P_BOUNDS[1])

    #intercept refitted at the shrunk slope, so the model still passes through
    #the weighted centroid of the pilots
    return my + p * mx, p


def predictSubSteps(results, target, nSamples):
    """(substeps predicted to hit target, fitted p) from the error model."""
    logC, p = fitErrorModel(results, nSamples)
    return math.exp((logC - math.log(impliedError(target))) / p), p


def resolvableWidth(hi, p, target, nSamples):
    """Bracket width below which pilot noise cannot tell two counts apart.

    dP/dlog(n) = p*x/(sqrt(pi)*exp(x^2)) with x = erfcinv(P), so pilots with
    standard error sigma localise the crossing to no better than
    dlog(n) = sigma / (dP/dlog n). Narrowing past that is measuring noise, not
    physics -- which is what made the old fixed hi//10 rule spend four or five
    pilots resolving a 10% bracket against 2.5% acceptance noise.
    """
    x = erfcinv(min(max(target, 1e-6), 1.0 - 1e-6))
    dPdLogN = p * x / (math.sqrt(math.pi) * math.exp(x * x))
    frac = pilotStdErr(target, nSamples) / max(dPdLogN, 1e-6)
    return max(1, math.ceil(hi * min(frac, 0.5)))


def tuneSubSteps(cfg):
    """
    Finds the smallest substep count meeting the target acceptance.

    Every pilot is read as a measurement of the integrator's energy error rather
    than as a point on an acceptance curve: <dH> = C * n^-p inverted out of
    P = erfc(sqrt(<dH>/2)) (see the model above). The next count to try is then
    the one the model says lands on the target, which needs no second point to
    get started — the first pilot already predicts an answer, and on the logged
    tunes that first prediction is typically within 10% of the count eventually
    chosen. This replaces a finite-difference secant that could not extrapolate
    until it had two points and fell back to doubling whenever pilot noise made
    its slope non-positive (which is what sent one m=0.01 Nx=48 tune to 228
    substeps). Growth is capped at 3n per step and never exceeds maxSubSteps,
    which is itself tried before giving up.

    The count returned is always one whose acceptance was measured and met the
    target, and unless confirmSteps is 0 it is then re-measured on a seed the
    search never saw. That check is not redundant: every search pilot reuses
    randSeed (common random numbers, which makes the differences between counts
    much less noisy than the acceptances themselves), so a seed that happens to
    read high would otherwise carry that luck straight into production. A
    confirmation that falls short is folded back in as an ordinary measurement
    and the search resumes from it.
    """
    target = cfg['targetAcceptance']
    maxN   = cfg['maxSubSteps']
    nSamp  = pilotSamples(cfg)
    results = {}                    # substeps -> measured acceptance
    pilots  = 0

    def measure(n):
        nonlocal pilots
        pilots += 1
        acc = pilotAcceptance(cfg, n)
        _, p = predictSubSteps({**results, n: acc}, target, nSamp)
        tqdm.write(f"  substeps {n:4d}: acceptance {acc:.2f}   (p={p:.1f})")
        results[n] = acc
        return acc

    def confirm(n):
        """Re-measure a tuned count on a seed the search never saw.

        A shortfall inside one standard error is pilot noise, not a real miss,
        so only a clear shortfall rejects. A rejection is recorded in `results`
        as an ordinary (failing) measurement, which both sharpens the model and
        re-brackets the search.
        """
        nonlocal pilots
        if not cfg['confirmSteps']:
            return True
        pilots += 1
        nConf = pilotSamples(cfg, cfg['confirmSteps'])
        #offset well clear of the per-chain seeds randSeed+i that production uses
        acc = pilotAcceptance(cfg, n, pilotSteps=cfg['confirmSteps'],
                              seed=cfg['randSeed'] + 9973)
        err = pilotStdErr(acc, nConf)
        if acc + err >= target:
            tqdm.write(f"  confirmed {n} substeps: acceptance {acc:.3f} +- {err:.3f} "
                       f"(target {target})")
            return True
        tqdm.write(f"  !! {n} substeps confirmed at {acc:.3f} +- {err:.3f}, short of "
                   f"{target}; continuing")
        results[n] = acc            # the independent estimate replaces the pilot's
        return False

    def bracket():
        """(largest failing count, smallest passing count) over all pilots."""
        lo = hi = None
        for n, acc in results.items():
            if acc >= target:
                hi = n if hi is None else min(hi, n)
            else:
                lo = n if lo is None else max(lo, n)
        return lo, hi

    n = min(cfg['startSubSteps'], maxN)
    while pilots < cfg['maxPilots']:
        measure(n)

        #settle on the next count to measure. A rejected confirmation demotes
        #the count it tested to a failing point, so this re-decides rather than
        #re-measuring what just failed.
        while True:
            lo, hi = bracket()
            nMax = max(results)
            guess, p = predictSubSteps(results, target, nSamp)
            nextN = max(1, math.ceil(guess))

            if hi is None:
                if nMax >= maxN:
                    raise RuntimeError(
                        f"acceptance {target} not reached by maxSubSteps={maxN} "
                        f"(the error model fits p={p:.1f} and predicts ~{nextN} "
                        f"substeps; raise maxSubSteps past that, or lower "
                        f"targetAcceptance)")
                n = min(max(nextN, nMax + 1), 3*nMax, maxN)
                break

            #a passing count is known: move only if the model says a materially
            #cheaper one exists, the bracket still has room for it, and that
            #room is wide enough for the pilots to actually resolve
            cand = min(nextN, hi - 1)
            if lo is not None:
                cand = max(cand, lo + 1)
            settled = (lo is not None
                       and hi - lo <= resolvableWidth(hi, p, target, nSamp))
            if (not settled and nextN < 0.95*hi
                    and cand < hi and (lo is None or cand > lo)):
                n = cand
                break

            if pilots >= cfg['maxPilots'] or confirm(hi):
                return hi

    lo, hi = bracket()
    if hi is not None:
        tqdm.write(f"  !! pilot budget ({cfg['maxPilots']}) exhausted; using {hi}")
        return hi

    #nothing reached the target. Fall back to the model's prediction rather than
    #aborting: an ensemble a few percent under target is usable, a dead scan is
    #not (and `set -e` in the scan scripts turns one abort into every remaining
    #volume). maxSubSteps stays a hard ceiling -- being unable to reach the
    #target below it is a configuration error, and that still raises above.
    guess, _ = predictSubSteps(results, target, nSamp)
    fallback = min(max(math.ceil(guess), max(results)), maxN)
    tqdm.write(f"  !! pilot budget ({cfg['maxPilots']}) exhausted with nothing reaching "
               f"{target} (best measured {max(results.values()):.2f}); falling back to "
               f"the model's {fallback} substeps")
    return fallback


def main(inputPath):
    cfg = loadInput(inputPath)

    #checked up front: saveEnsemble would also refuse, but only after the chains have run
    if os.path.exists(cfg['outputFile']):
        sys.exit(f"{cfg['outputFile']} already exists; move it or change outputFile")

    if cfg['numSubSteps'] is not None:
        numSubSteps = cfg['numSubSteps']
        print(f"using {numSubSteps} substeps (given in input file)")
        #a hand-set count gets the same acceptance check a tuned one does, so a
        #stale number copied between input files cannot quietly run a whole
        #ensemble at the wrong acceptance
        if cfg['confirmSteps']:
            acc = pilotAcceptance(cfg, numSubSteps, pilotSteps=cfg['confirmSteps'],
                                  seed=cfg['randSeed'] + 9973)
            err = pilotStdErr(acc, pilotSamples(cfg, cfg['confirmSteps']))
            note = "" if acc + err >= cfg['targetAcceptance'] else "  !! below target"
            print(f"  pilot acceptance {acc:.3f} +- {err:.3f} "
                  f"(target {cfg['targetAcceptance']}){note}")
    else:
        print(f"tuning substeps for acceptance >= {cfg['targetAcceptance']}:")
        numSubSteps = tuneSubSteps(cfg)
        print(f"using {numSubSteps} substeps")

    stepsPerChain = -(-cfg['metroSteps'] // cfg['nChains'])   # ceil division
    print(f"running {cfg['nChains']} chains x ({cfg['burnIn']} burn-in + {stepsPerChain} configs)"
          f" on the {cfg['backend']}")

    getRunner(cfg['backend']).runExperiment(
        cfg['outputFile'],
        beta=cfg['beta'], fMass=cfg['fMass'], aSpacing=cfg['aSpacing'],
        Nx=cfg['dimx'], Nt=cfg['dimt'],
        metroSteps=cfg['metroSteps'], numSubSteps=numSubSteps,
        tunneling=cfg['tunneling'], cores=cfg['nCores'], chains=cfg['nChains'],
        perChainBurnIn=cfg['burnIn'],
        cgRtolForce=cfg['cgRtolForce'], cgRtolAction=cfg['cgRtolAction'],
        fermionAction=cfg['fermionAction'], dim5=cfg['dim5'], M5=cfg['M5'],
        coldStartForce=cfg['coldStartForce'],
        randSeed=cfg['randSeed'],
    )

    print(f"saved {cfg['metroSteps']} configurations to {cfg['outputFile']}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("usage: python run_sim.py <input.toml>")
    main(sys.argv[1])
