"""
Gauge configuration generation driven by a TOML input file:

    python run_sim.py inputs/example.toml

Runs nChains independent HMC chains in parallel, discards the per-chain burn-in,
merges the thermalized configurations, and writes them to an HDF5 ensemble file.
The number of HMC substeps is either given explicitly (numSubSteps) or tuned
automatically by scanning pilot chains until the target metropolis acceptance
rate is reached. See inputs/example.toml for all recognized fields.
"""

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
        #sea quark discretization: "wilson" (default) or "dwf" (needs dim5, M5)
        'fermionAction': raw['physics'].get('fermionAction', 'wilson'),
        'dim5': raw['physics'].get('dim5'),
        'M5':   raw['physics'].get('M5'),
    }
    if cfg['metroSteps'] is None:
        raise KeyError(f"{path} sets neither run.metroSteps nor run.targetConfigs")
    if cfg['fermionAction'] == 'dwf' and (cfg['dim5'] is None or cfg['M5'] is None):
        raise KeyError(f"{path} sets fermionAction='dwf' but not physics.dim5 and physics.M5")
    cfg['nChains'] = run.get('nChains', cfg['nCores'])

    sub = raw.get('substeps', {})
    cfg['numSubSteps']      = sub.get('numSubSteps')        # None -> tune automatically
    cfg['targetAcceptance'] = sub.get('targetAcceptance', 0.8)
    cfg['pilotSteps']       = sub.get('pilotSteps', 100)
    cfg['startSubSteps']    = sub.get('startSubSteps', 5)
    cfg['maxSubSteps']      = sub.get('maxSubSteps', 500)

    return cfg


def pilotAcceptance(cfg, numSubSteps):
    """Acceptance rate of a short pilot chain, measured on its second half
    so the cold-start transient does not bias the estimate."""
    accept, _ = experiment.acceptanceFractions(
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
    Scans substep counts until the target acceptance is reached: doubling to
    bracket, then bisecting to the smallest passing count (within ~10%).
    """
    target = cfg['targetAcceptance']

    lo = None                       # last failing count
    hi = cfg['startSubSteps']
    while hi <= cfg['maxSubSteps']:
        acc = pilotAcceptance(cfg, hi)
        tqdm.write(f"  substeps {hi:4d}: acceptance {acc:.2f}")
        if acc >= target:
            break
        lo, hi = hi, hi*2
    else:
        raise RuntimeError(f"acceptance {target} not reached by {cfg['maxSubSteps']} substeps")

    if lo is None:
        return hi

    #bisect the bracket [lo (fail), hi (pass)] down to ~10% granularity
    while hi - lo > max(1, hi//10):
        mid = (lo + hi)//2
        acc = pilotAcceptance(cfg, mid)
        tqdm.write(f"  substeps {mid:4d}: acceptance {acc:.2f}")
        if acc >= target:
            hi = mid
        else:
            lo = mid

    return hi


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
    print(f"running {cfg['nChains']} chains x ({cfg['burnIn']} burn-in + {stepsPerChain} configs)")

    experiment.runExperiment(
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
