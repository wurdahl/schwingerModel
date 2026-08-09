"""
Gauge configuration generation driven by a TOML input file:

    python run_sim.py inputs/example.toml

Runs nChains independent HMC chains in parallel, discards the per-chain burn-in,
merges the thermalized configurations, and writes them to an HDF5 ensemble file.
The number of HMC substeps is either given explicitly (numSubSteps) or tuned
automatically by scanning pilot chains until the target metropolis acceptance
rate is reached. See inputs/example.toml for all recognized fields.
"""

import math
import os
import sys
import tomllib

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

    return cfg


def getRunner(backend):
    """The module the run routes through. The gpu module is imported lazily because
    importing jax initializes the gpu (and preallocates most of its memory), which
    a cpu run must never do."""
    if backend == 'gpu':
        from schwingerModel import experiment_gpu
        return experiment_gpu
    return experiment


def pilotAcceptance(cfg, numSubSteps):
    """Acceptance rate of a short pilot chain, measured on its second half
    so the cold-start transient does not bias the estimate."""
    accept, _ = getRunner(cfg['backend']).acceptanceFractions(
        beta=cfg['beta'], fMass=cfg['fMass'], aSpacing=cfg['aSpacing'],
        Nx=cfg['dimx'], Nt=cfg['dimt'],
        metroSteps=cfg['pilotSteps'], numSubSteps=numSubSteps,
        tunneling=cfg['tunneling'],
        cgRtolForce=cfg['cgRtolForce'], cgRtolAction=cfg['cgRtolAction'],
        fermionAction=cfg['fermionAction'], dim5=cfg['dim5'], M5=cfg['M5'],
        seed=cfg['randSeed'],
        tqdmPosition=1,   #keep the pilot bar below the tuning report lines
    )
    return accept


def tuneSubSteps(cfg):
    """
    Finds the smallest substep count meeting the target acceptance.

    While no passing count is known, each step extrapolates to the target along
    the measured acceptance-vs-substeps slope (a secant/Newton step), clamped to
    [n+1, 2n] so one noisy slope can neither stall nor run away, and never past
    maxSubSteps — the cap itself is tried before giving up. Once a fail/pass
    bracket exists, secant steps through its endpoints (midpoint fallback)
    narrow it to ~10% granularity. Pilot acceptances are noisy; the tolerance
    absorbs that.
    """
    target = cfg['targetAcceptance']
    maxN = cfg['maxSubSteps']
    results = {}                    # substeps -> measured acceptance
    order = []                      # evaluation order, for the growth-phase secant

    def measure(n):
        acc = pilotAcceptance(cfg, n)
        tqdm.write(f"  substeps {n:4d}: acceptance {acc:.2f}")
        results[n] = acc
        order.append(n)
        return acc

    def secant(n1, n2, fallback):
        """Substep count where the line through both measurements hits target."""
        if n1 is None or n1 == n2:
            return fallback
        slope = (results[n2] - results[n1]) / (n2 - n1)
        if slope <= 0:              # pilot noise: acceptance rises with substeps
            return fallback
        return n2 + (target - results[n2]) / slope

    n = min(cfg['startSubSteps'], maxN)
    lo, hi = None, None             # largest failing / smallest passing count
    while True:
        acc = measure(n)
        if acc >= target:
            hi = n if hi is None else min(hi, n)
        else:
            lo = n if lo is None else max(lo, n)

        if hi is not None:
            if lo is None or hi - lo <= max(1, hi//10):
                return hi
            #narrow the bracket: secant through its endpoints, midpoint fallback
            guess = secant(lo, hi, (lo + hi)/2)
            n = min(max(math.ceil(guess), lo + 1), hi - 1)
        else:
            if n >= maxN:
                raise RuntimeError(f"acceptance {target} not reached by {maxN} substeps")
            #grow: secant through the last two points, doubling fallback
            prev = order[-2] if len(order) > 1 else None
            guess = secant(prev, n, 2*n)
            n = min(max(math.ceil(guess), n + 1), 2*n, maxN)


def main(inputPath):
    cfg = loadInput(inputPath)

    #checked up front: saveEnsemble would also refuse, but only after the chains have run
    if os.path.exists(cfg['outputFile']):
        sys.exit(f"{cfg['outputFile']} already exists; move it or change outputFile")

    if cfg['numSubSteps'] is not None:
        numSubSteps = cfg['numSubSteps']
        print(f"using {numSubSteps} substeps (given in input file)")
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
        randSeed=cfg['randSeed'],
    )

    print(f"saved {cfg['metroSteps']} configurations to {cfg['outputFile']}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("usage: python run_sim.py <input.toml>")
    main(sys.argv[1])
